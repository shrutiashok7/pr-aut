# PR-AUT

This tool automates GitHub Pull Request creation using LLM (Groq).

## Flow
- Get code changes (git diff)
- Create new branch: `UT-gen-<timestamp>`
- Commit generated tests (in `tests/`)
- Generate PR title & description using Groq API
- Push branch and raise Pull Request via GitHub API

## Setup
1. Install dependencies:

2. Add your `.env` file with API tokens.

3. Run:

#poetry run python main.py --generate-pr