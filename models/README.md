# vLLM Models Directory

Place your GGUF model files here. They will be mounted into the vLLM container at `/models`.

## Example

```bash
# Download a model
huggingface-cli download TheBloke/Llama-2-7B-GGUF llama-2-7b.Q4_K_M.gguf --local-dir .

# Start vLLM with this model
docker compose --profile vllm up -d
```
