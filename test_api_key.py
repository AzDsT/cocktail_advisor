import requests

url = "http://127.0.0.1:8000/ask"
payload = {
    "question": "What is the history of the Margarita?"
}
response = requests.post(url, json=payload)
print(response.json())