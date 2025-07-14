import requests

# Client script to test the FastAPI distance endpoint by sending sentence pairs and receiving their embedding distance

data = {
    "sentence1": "hello",
    "sentence2": "hi there",
    "model_choice": "all-MiniLM-L6-v2",
    "distance_metric": "cosine"
}

response = requests.post("http://127.0.0.1:8000/dist", json=data)
print(response.json())
