# API-Free Public Version

The public repository is designed to run without paid external API access.

## Disabled By Default

- Gemini provider calls.
- DeepSeek provider calls.
- OpenAI provider calls.
- Any script path requiring private `.env` files or provider credentials.

## What Changed For Public Release

- A root `.gitignore` excludes `.env`, checkpoints, raw dataset exports, virtual environments, caches, and large model artifacts.
- `.env.example` contains placeholders only.
- External provider runners and raw provider artifacts are excluded from the public runnable core.
- Public summary scripts read saved outputs instead of calling external providers.
- Documentation labels external LLM use as a design extension only.

## Secret Handling

No public workflow loads `.env` files or sends external API requests. If you intentionally rerun historical API experiments, keep keys in your private shell or ignored `.env` file and rotate them before publication if they were ever exposed.
