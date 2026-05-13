# NixOrb Configuration

Config file: `~/.config/nixorb/config.toml`
Created on first run from `config/default.toml`.

## CLI Config Commands

```bash
nixorb config                        # show all settings
nixorb config llm_model              # read one key
nixorb config llm_model gpt-4o-mini  # set a key
nixorb config-gui                    # open Settings window
```

## Key Settings

| Key | Default | Description |
|---|---|---|
| `hotkey` | `Ctrl+Alt+Space` | Global activation hotkey |
| `llm_backend` | `huggingface` | `huggingface` / `openai` / `ollama` / `local` |
| `llm_model` | `torphix/stablelm-2-glados-v1` | HF repo ID or model name |
| `hf_token` | `""` | HuggingFace token for gated models |
| `tts_backend` | `huggingface` | `huggingface` / `openai` / `piper` / `glados` |
| `vision_model` | `thwri/CogFlorence-2.2-Large` | Screen captioning model |
| `vlm_model` | `Qwen/Qwen3.5-4B` | Full vision+LLM model |
| `web_search_enabled` | `true` | Auto web search on relevant queries |
| `wake_word_enabled` | `false` | OpenWakeWord always-on detection |

## Switching Backends

**OpenAI:**
```bash
nixorb config llm_backend openai
nixorb config openai_api_key sk-xxxxx
nixorb config llm_model gpt-4o-mini
```

**Local GGUF (llama.cpp):**
```bash
nixorb config llm_backend local
nixorb config local_model_path /path/to/model.Q4_K_M.gguf
```

**Ollama:**
```bash
ollama pull mistral
nixorb config llm_backend ollama
nixorb config llm_model mistral
```
