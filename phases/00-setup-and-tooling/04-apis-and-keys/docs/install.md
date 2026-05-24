# Install Dependencies

## JavaScript / TypeScript

Install the Anthropic SDK:

```bash
npm install @anthropic-ai/sdk
```

Install TypeScript + tsx for running `.ts` files:

```bash
npm install -D typescript tsx
```

## Ollama

Install Ollama:

```bash
brew install ollama
```

Start Ollama:

```bash
ollama serve
```

Pull a local model:

```bash
ollama pull qwen3-coder:30b
```

## Run the Script

```bash
npx tsx code/local_api_call.ts
```
