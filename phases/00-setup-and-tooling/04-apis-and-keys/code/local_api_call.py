import anthropic

client = anthropic.Anthropic(
    base_url="http://localhost:11434",
    api_key="ollama",
)

response = client.messages.create(
    model="qwen3-coder:30b",
    max_tokens=256,
    messages=[{"role": "user", "content": "What is a neural network in one sentence?"}],
)

print(response.content[0].text)
