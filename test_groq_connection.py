"""Verify a local GROQ_API_KEY can call Groq's Llama 3.3 model."""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv
from groq import Groq, GroqError


MODEL = "openai/gpt-oss-20b"


def main() -> int:
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "your_groq_api_key_here":
        print("GROQ_API_KEY is missing. Set it in .env before running this script.", file=sys.stderr)
        return 1

    try:
        response = Groq(api_key=api_key).chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": "Reply with exactly: connection verified"}],
            temperature=0,
            max_completion_tokens=16,
        )
    except GroqError as error:
        print(f"Groq connection failed: {error}", file=sys.stderr)
        return 1

    message = response.choices[0].message.content or ""
    print(f"Groq connection verified with {MODEL}: {message.strip()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
