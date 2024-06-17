#!/usr/bin/env python
# to ignore the deprecation warnings
import warnings
from langchain._api import LangChainDeprecationWarning
warnings.simplefilter("ignore", category=LangChainDeprecationWarning)

import uuid
import chromadb
from langchain import hub
from langchain_community.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain.embeddings import GPT4AllEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain.chains import RetrievalQA
from nemoguardrails import RailsConfig
from nemoguardrails.integrations.langchain.runnable_rails import RunnableRails

#load the LLM
def load_llm():
    llm = Ollama(
        model = "llama3",
        base_url = "http://localhost:11434",
        temperature = 0,
        verbose = True,
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()]),
    )
    return llm

# Create a ChatPromptTemplate 
def create_prompt_template():
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a research assistant specializing in question-answering tasks. "
            "Use the provided context to answer the question concisely. "
            "If you don't know the answer, simply state that you don't know. "
            "Keep your response to a maximum of three sentences. "
            "Keep following 3 instructions in the structured format. "
            "1. Summarize the main points in bullet points. This is the highest priority. "
            "2. Provide a detailed description with sufficient detail for understanding, without unnecessary elaboration. "
            "3. Include any additional relevant information, if available, but keep it brief and optional. "
        ),
        (
            "user", 
            "Question: {question}\n" 
            "Context: {context}\n"
            "Answer: ",
        ),
        (
            "assistant", 
            "**Main Points:**\n"
            "**Detailed Description:**\n"
            "**Additional Information:**\n"
        )
    ])
    return prompt

# Vector DB functions
DATA_PATH="/mnt/c/Users/beene/Downloads/papers/"
DB_PATH = "./chroma"

def create_documents():
    '''Upload the PDFs in the DATA_PATH directory'''
    loader = PyPDFDirectoryLoader(DATA_PATH, silent_errors = False)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 50)
    texts = text_splitter.split_documents(documents)
    return texts

def create_embeddings(documents):
    '''Create embeddings for the documents in the DATA_PATH directory'''
    embedder = GPT4AllEmbeddings()
    return [embedder.embed_query(doc.page_content) for doc in documents]

def load_client():
    '''Load the client'''
    print("Loading client ... \n")
    return chromadb.PersistentClient(path = DB_PATH)

def clear_db(client):
    list_collections = client.list_collections()
    if len(list_collections):
        for i in list_collections:
            client.delete_collection(i.dict()["name"])

def create_vector_db(client):
    '''Create a vector db from the documents in the DATA_PATH directory'''
    # Make original paper db and duplicated db for search purposes
    # , which are saved to the DB_PATH directory
    _ = client.create_collection(
        name = "Original",
        metadata={"hnsw:space": "cosine"},
    )
    _ = client.create_collection(
        name = "Search",
    )

def add_documents_to_db(client):
    # Load the vector db client
    collection = client.get_collection("Original")

    # Create documents and embeddings
    documents = create_documents()
    embeddings = create_embeddings(documents)

    # Add the documents and embeddings to the collection
    collection.add(
        ids = [str(uuid.uuid4()) for _ in documents],
        documents = [doc.page_content for doc in documents],
        metadatas = [doc.metadata for doc in documents],
        embeddings = embeddings
    )

def implementation_db(client):
    '''Implementation of the above functions'''
    print("Create vector database (it takes a while depending on dataset)\n")
    clear_db(client) # Clean up the existing db
    create_vector_db(client)
    add_documents_to_db(client)
    print("Done.\n\n")

def duplicate_db(client):
    '''Duplicate the existing vector db for iterative search purposes'''
    print("Make db duplicates for a search purpose\n")
    # Clear the search db
    client.delete_collection("Search")
    print("Done.\n\n")

    # Load the vector db client
    collection = client.get_collection("Original")
    collection2 = client.create_collection(
        name = "Search", 
        metadata={"hnsw:space": "cosine"},
    )
    
    # Dupliacte a collection
    total_documents_count = collection.count()
    batch_size = 10
    for i in range(0, total_documents_count, batch_size):
        batch = collection.get(
            include=["metadatas", "documents", "embeddings"],
            limit=batch_size,
            offset=i
        )
        collection2.add(
            ids=batch["ids"],
            documents=batch["documents"],
            metadatas=batch["metadatas"],
            embeddings=batch["embeddings"]
        )    

def load_vector_db(client, collection_name):
    '''Load the vector db for chaining purposes'''
    return Chroma(
        client = client,
        persist_directory = DB_PATH,
        collection_name = collection_name,
        embedding_function = GPT4AllEmbeddings(),
        collection_metadata={"hnsw:space": "cosine"},
    )

def extract_source_document(response):
    '''Extract the source document from the response'''
    source_document = response.get("source_documents")[0].dict()
    return source_document["metadata"]["source"]

def delete_searched_document(client, response):
    '''Delete the searched document from the db'''
    collection = client.get_collection("Search")
    collections = collection.get() # get a dict db

    # find the id
    source_document = extract_source_document(response)
    ids = collections["ids"]
    ids_to_delete = []
    for i in range(len(ids)):
        if collections["metadatas"][i]["source"] == source_document:
            ids_to_delete.append(ids[i])

    collection.delete(ids = ids_to_delete)


# Retrieval QA functions
# Contextual compression retriever
def get_retrieval(client):
    compressor = FlashrankRerank(top_n = 1)
    chroma_db = load_vector_db(client, "Search")
    return ContextualCompressionRetriever(
        base_compressor = compressor, 
        base_retriever = chroma_db.as_retriever()
    )

def retrieval_qa_chain(client, llm, prompt):
    retriever = get_retrieval(client)
    # return RetrievalQA.from_chain_type(
    #     # (llm | guardrails),
    #     llm,
    #     retriever = retriever,
    #     chain_type_kwargs = {"prompt": prompt},
    #     return_source_documents = True,
    # )

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain  

def check_relevance_score(query):
    '''Check the relevance score of the query'''
    threshold = 0.5
    retrieved_documents = retriever.invoke(query)
    if len(retrieved_documents):
        return retrieved_documents[0].metadata["relevance_score"] >= threshold
    else:
        return False