import json
import urllib.request

# Anthropic-compatible API
url = "http://localhost:11434/v1/messages"

headers = {
    "Content-Type": "application/json",
    "x-api-key": "ollama",
    "anthropic-version": "2023-06-01",
}

body = json.dumps(
    {
        "model": "qwen3-coder:30b",
        "max_tokens": 256,
        "messages": [
            {"role": "user", "content": "What is a neural network in one sentence?"}
        ],
    }
).encode()

req = urllib.request.Request(url, data=body, headers=headers, method="POST")

with urllib.request.urlopen(req) as resp:
    result = json.loads(resp.read())
    print(result["content"][0]["text"])
