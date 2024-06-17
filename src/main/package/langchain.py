#!/usr/bin/env python
# to ignore the deprecation warnings and info
import logging
import warnings
from langchain._api import LangChainDeprecationWarning
warnings.simplefilter("ignore", category=LangChainDeprecationWarning)
warnings.filterwarnings('ignore')
logging.disable(logging.INFO)

# to hide automatic prints
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout    

# import the necessary packages
import uuid
import sys, os
import chromadb
import numpy as np
from pathlib import Path
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
from nemoguardrails import RailsConfig, LLMRails

# PATHs
DATA_PATH = ""
DB_PATH = "./chroma"

# Load or make the client
print("Loading client ... \n")
client = chromadb.PersistentClient(path = DB_PATH)

# Preload the LLM
llm = Ollama(
    model = "llama3",
    base_url = "http://localhost:11434",
    temperature = 0,
    verbose = True,
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()]),
)

# Nemo Guardrails
def load_guardrails():
    config = RailsConfig.from_path("./config/config.yml")
    rails = LLMRails(config, llm = llm)
    return rails

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
            "1. Summarize the main points in bullet points, using •. This is the highest priority. "
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

def count_pdfs():
    import os
    count = 0
    for root, dirs, files in os.walk(DATA_PATH):
        count += len([fn for fn in files if fn.endswith(".pdf")])
    return count

# Vector DB functions
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

def clear_db():
    list_collections = client.list_collections()
    if len(list_collections):
        for i in list_collections:
            client.delete_collection(i.dict()["name"])

def create_vector_db():
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

def add_documents_to_db():
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

def implementation_db():
    '''Implementation of the above functions'''
    # Load LM    
    print(
        "\nCreate vector database ..."
        "It could take a while depending on your dataset."
    )
    with HiddenPrints():
        clear_db() # Clean up the existing db
        create_vector_db()
        add_documents_to_db()
    print("Done.\n\n")

def load_vector_db(collection_name):
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

def delete_searched_document(response):
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
def get_retrieval():
    compressor = FlashrankRerank(top_n = 1)
    chroma_db = load_vector_db("Search")
    return ContextualCompressionRetriever(
        base_compressor = compressor, 
        base_retriever = chroma_db.as_retriever()
    )

def get_chain(rail_llm):
    guardrails = load_guardrails()
    prompt = create_prompt_template()
    retriever = get_retrieval()
    return RetrievalQA.from_chain_type(
        rail_llm,
        retriever = retriever,
        chain_type_kwargs = {"prompt": prompt},
        return_source_documents = True,
    )

def check_relevance_score(retriever, query):
    '''Check the relevance score of the query'''
    threshold = 0.5
    retrieved_documents = retriever.invoke(query)
    if len(retrieved_documents):
        return retrieved_documents[0].metadata["relevance_score"] >= threshold
    else:
        return False

# Make the response pretty
def remove_empty_lines(result):
    copy_result = result.copy()
    for i in result:
        if len(i) == 0:
            copy_result.remove(i)
    indices = []
    for i in range(len(copy_result)):
        if copy_result[i].endswith("**") and i != 0:
            indices.append(i)
    for i in indices[::-1]:
        copy_result.insert(i, "")         
    return copy_result

def print_bunch():
    return print(
        "---------------------------------------"
        "---------------------------------------"
        "---------------------------------------"
    )

def pprint_response(responses, response):
    result = remove_empty_lines(response["result"].split("\n")[1:])
    file_path = response["source_documents"][0].metadata["source"]
    file_name = Path(file_path).name
    print_bunch()
    print(f"\nDocuments #{len(responses)}")
    print(f"The source document: {file_name}\n\n")
    _ = [print(phrase) for phrase in result]
    print_bunch()

# Main functions
def ask_user_to_create_db():
    while True:
        user_input = input("Do you want to create a new database? (y/n): ")
        if user_input == "n":
            break
        elif user_input == "y":
            global DATA_PATH
            DATA_PATH = input("Enter the path to the directory containing PDFs: ")
            implementation_db()
            break
        print("Invalid input. Please enter \'y\' or \'n\'.")

def ask_how_many_documents():
    # print(f"\n***Total documents: {count_pdfs()}***")
    while True:
        iter_num = input(
            f"How many documents do you want to search? (<= {count_pdfs()}): " 
        )
        if iter_num.isdigit() and int(iter_num) <= count_pdfs():
            return int(iter_num)
        print("Invalid input. Please enter a number.")

def prepare_db():
    '''Duplicate the existing vector db for iterative search purposes'''
    print("\nPreparing a database for a search purpose ... ")
    # Clear the search db
    client.delete_collection("Search")

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
    print("Done.\n")

def response_loop(iter_num):
    assert iter_num <= count_pdfs()
    '''Iterative search for documents'''
    # params
    rails = load_guardrails()
    retriever = get_retrieval()
    chain = get_chain(rails.llm)
    
    # Implement the search loop
    responses = []
    query = input("\nWhat is your question?: ")
    for _ in range(iter_num):
        # Check if the search db is empty
        if client.get_collection("Search").count() == 0:
            print("\n=======================================")
            print("No documents left to search.")
            print("=======================================\n")
            break

        # Check the relevance score of the query
        if check_relevance_score(retriever, query):
            # Get response
            with HiddenPrints():
                response = chain.invoke(query)
            
            # Print the response (Custom)
            responses.append(response)
            pprint_response(responses, response)

            # Delete the returned document from the db
            delete_searched_document(response)
        else: 
            # Quit the search loop if the relevance score is below the threshold
            print("\n=======================================")
            print("A searched document is not relevant.")
            print("=======================================\n")
            break
    print("\n==================================")    
    print(f"Total number of search is {len(responses)}.")
    print("Searching is done.")
    print("==================================\n")
    return responses

def ask_if_more_to_ask():
    while True:
        user_input = input("Do you want to ask more questions? (y/n): ")
        if user_input == "n":
            print("\n\nGoodbye!")
            quit()
        elif user_input == "y":
            return
        print("Invalid input. Please enter \'y\' or \'n\'.")