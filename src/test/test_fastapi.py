from fastapi import FastAPI
import asyncio
import nest_asyncio
from langserve import add_routes
from langchain.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

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

llm = load_llm()

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="Spin up a simple api server using Langchain's Runnable interfaces",
)

add_routes(
    app,
    llm,
    path="/llm",
)

nest_asyncio.apply()

async def main():
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)

if __name__ == "__main__":
    asyncio.run(main())