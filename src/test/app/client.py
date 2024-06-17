import requests

class LangserveClient:
    def __init__(self, base_url):
        self.base_url = base_url

    def get_root(self):
        response = requests.get(f"{self.base_url}/")
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": response.status_code}

    def process_text(self, text):
        response = requests.post(f"{self.base_url}/process", json={"text": text})
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": response.status_code}

if __name__ == "__main__":
    client = LangserveClient("http://127.0.0.1:8000")
    
    # Test the root endpoint
    root_response = client.get_root()
    print("Root Response:", root_response)
    
    # Test the text processing endpoint
    text_to_process = "This is a test text."
    process_response = client.process_text(text_to_process)
    print("Process Response:", process_response)