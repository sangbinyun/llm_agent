# llm_agent
A research assistant to find and abstract as many relevant papers as possible using Ollama & Llama3, LangChain, and Nemo-Guardrails.

1. Install all packages in pyproject.toml with poetry
2. Clone this repository
3. Implement src/main/main.py
4. Follow the instructions. 
    - Create DB
    - Define a path where your pdfs are stored
    - Set how much documents you want to get

Description: 
- This LLM application will look over all of your pdfs and return one document at a time.
- The above process will be done in a iterative way.
