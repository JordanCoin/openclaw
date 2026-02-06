---
summary: "Use OpenRouter's unified API to access many models in OpenClaw"
read_when:
  - You want a single API key for many LLMs
  - You want to run models via OpenRouter in OpenClaw
title: "OpenRouter"
---

# OpenRouter

OpenRouter provides a **unified API** that routes requests to many models behind a single
endpoint and API key. It is OpenAI-compatible, so most OpenAI SDKs work by switching the base URL.

## CLI setup

```bash
openclaw onboard --auth-choice apiKey --token-provider openrouter --token "$OPENROUTER_API_KEY"
```

## Config snippet

Put your API key in the environment (recommended) or in `env` in config:

```bash
export OPENROUTER_API_KEY="sk-or-..."
```

```json5
{
  agents: {
    defaults: {
      model: { primary: "openrouter/google/gemini-2.5-flash" },
      models: {
        "openrouter/google/gemini-2.5-flash": {},
        "openrouter/openrouter/free": {},
      },
    },
  },
}
```

## Free tier and Gemini

- **Free models:** Use the Free Models Router with one API key: set `primary` to `openrouter/openrouter/free`. OpenRouter picks a free model per request. See [OpenRouter free models](https://openrouter.ai/docs/guides/guides/free-models-router-playground).
- **Gemini via OpenRouter:** Use model refs like `openrouter/google/gemini-2.5-flash`, `openrouter/google/gemini-2.5-pro`, or `openrouter/google/gemini-2.0-flash-001`. No separate Google API key needed; only `OPENROUTER_API_KEY` is used.
- **Auto (cost-aware):** `openrouter/openrouter/auto` chooses a model by prompt and cost.

## Notes

- Model refs are `openrouter/<provider>/<model>` (e.g. `openrouter/google/gemini-2.5-flash`).
- For more providers and options, see [Model providers](/concepts/model-providers).
- You can store the key securely: `openclaw auth set openrouter:default --key "sk-or-..."` after adding an `auth.profiles["openrouter:default"]` entry with `provider: "openrouter"` and `mode: "api_key"`.
