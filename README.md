# Visual Prompt Studio

A local Streamlit workbench for planning image prompts with DeepSeek and generating images through an OpenAI-compatible image API.

## Features

- Expand an initial idea into editable visual directions.
- Review and select scene imagery and optional style candidates before prompt development.
- Build concise, structured prompts using a searchable visual-term knowledge base.
- Generate images, revise the final prompt, and keep project history locally.
- Add anatomy safeguards for human subjects and a restrained-detail mode for anime-style illustrations.

## Requirements

- Python 3.11 or newer
- A DeepSeek API key
- Access to an OpenAI-compatible image-generation API and its API key

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env` and add your own credentials and provider settings. Never commit `.env` or paste API keys into source files. The included `.env.example` contains placeholders only.

The image endpoint defaults to `https://xcode.best/v1` and model `gpt-image-2`; change `OPENAI_BASE_URL`, `OPENAI_IMAGE_MODEL`, and related variables in `.env` to match your provider. DeepSeek uses `https://api.deepseek.com` by default.

## Run

```powershell
python -m streamlit run app.py
```

Open the local address printed by Streamlit. Projects, generated images, and verification artifacts are stored under the ignored `workspace/` directory and remain on your machine.

## Tests

```powershell
python -X utf8 -m unittest test_knowledge test_openai_image test_app -q
```

Live DeepSeek checks are opt-in and make API requests:

```powershell
python -X utf8 test_deepseek.py --live --workflow
python -X utf8 test_deepseek.py --live --concise
```

Do not pass `--images` unless you intend to make a paid image-generation request.

## Knowledge Base

The curated runtime vocabulary is included at `data/visual_knowledge.json`. Build checkpoints, source reference files, request/response logs, and personal project data are not part of this repository. The normal application does not call the knowledge-building scripts.

Knowledge descriptions are AI-generated and have not been individually reviewed by subject-matter experts. Check important terminology before relying on it.

## Privacy

This repository ignores `.env`, local projects, generated images, request logs, source documents, and internal reports. Confirm `git status` before committing, and rotate any credential that may have been exposed elsewhere.
