#!/usr/bin/env python
import asyncio
import nest_asyncio
import chromadb
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langserve import add_routes
from typing import List
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.documents.base import Document
from functions import load_llm, load_client, implementation_db, duplicate_db, create_prompt_template, retrieval_qa_chain

def basic_implementation():
    # Load LM
    llm = load_llm()

    # Implement the DB to store documents
    client = load_client()
    # implementation_db(client)
    duplicate_db(client)

    # Create chain
    prompt = create_prompt_template()
    chain = retrieval_qa_chain(client, llm, prompt)

    return chain

class Output(BaseModel):
    # chat_history: List[Union[HumanMessage, AIMessage]]
    input: str
    context: List[Document]
    answer: str

def create_application(chain):
    # Implement applicagtion server
    app = FastAPI(
        title="LangChain Server",
        version="1.0",
        description="Research assistant agent using llama3",
    )

    add_routes(
        app,
        chain.with_types(output_type=Output),
        path="/llm",
    )

    @app.get("/")
    async def redirect_root_to_docs():
        return RedirectResponse("/docs")    

    # @app.get("/llm")
    # async def query_llm(question: str):
    #     try:
    #         response = await chain.run({"context": "Your context here", "question": question})
    #         return {"response": response}
    #     except Exception as e:
    #         raise HTTPException(status_code=500, detail=str(e))

    @app.on_event("shutdown")
    async def shutdown_event():
        print("Shutting down server...")

    return app

async def main():
    import uvicorn

    config = uvicorn.Config(app, host="localhost", port=8000)
    server = uvicorn.Server(config)    
    try:
        await server.serve()
    except KeyboardInterrupt:
        print("Server stopped by user")
    finally:
        await server.shutdown()
        # server.should_exit = True
        print("Server shutdown complete")        

if __name__ == "__main__":
    chain = basic_implementation()
    app = create_application(chain)
    nest_asyncio.apply()
    print("\nServer is running ... \n")
    asyncio.run(main())