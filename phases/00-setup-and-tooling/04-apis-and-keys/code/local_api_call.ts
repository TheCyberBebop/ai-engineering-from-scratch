import Anthropic from "@anthropic-ai/sdk";

async function main() {
  const client = new Anthropic({
    baseURL: "http://localhost:11434/",
    apiKey: "ollama",
  });

  const response = await client.messages.create({
    model: "qwen3-coder:30b",
    max_tokens: 256,
    messages: [
      {
        role: "user",
        content: "What is a neural network in one sentence?",
      },
    ],
  });

  console.log(response.content[0].type === "text" ? response.content[0].text : response.content[0]);
}

main().catch(console.error);
