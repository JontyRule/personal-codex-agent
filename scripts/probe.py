from __future__ import annotations

import os
import argparse
from typing import List

from dotenv import load_dotenv  # type: ignore
from openai import OpenAI

from rag.build_index import build_index
from rag.retriever import retrieve
from utils.loader import load_profile


def get_client() -> OpenAI:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set. See .env.example.")
    return OpenAI(api_key=key)


def compose_answer(question: str, mode_label: str, reflective: bool, temperature: float):
    # Simple non-UI composition for probing
    prof = load_profile()
    r = retrieve(question, top_k=4, prioritize_reflection=reflective)

    # Build prompts
    system = open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts", "system_base.txt"), "r", encoding="utf-8").read()
    mode_file = {
        "Interview": "mode_interview.txt",
        "Storytelling": "mode_storytelling.txt",
        "Fast Facts": "mode_fastfacts.txt",
        "Humble Brag": "mode_humblebrag.txt",
        "Reflective": "mode_reflective.txt",
    }[mode_label]
    mode_text = open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts", mode_file), "r", encoding="utf-8").read()

    context = "\n\n".join([
        f"[Source: {os.path.basename(x['source'])}#{x['heading']}]\n{x['text']}" for x in r["results"]
    ])

    sources_footer = " ".join([f"[Doc: {os.path.basename(x['source'])}#{x['heading']}]" for x in r["results"]])

    client = get_client()
    model = os.environ.get("OPENAI_MODEL", "gpt-4o")

    messages = [
        {"role": "system", "content": system},
        {"role": "system", "content": f"PROFILE\n{prof.to_prompt_block()}"},
        {"role": "system", "content": f"MODE\n{mode_text}"},
        {"role": "system", "content": f"CONTEXT\n{context}"},
        {"role": "user", "content": question},
    ]

    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )

    answer = resp.choices[0].message.content

    return r, (answer or ""), sources_footer


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Probe the Personal Codex Agent")
    parser.add_argument("question", type=str, help="Question to ask")
    parser.add_argument("--mode", default="Interview", choices=["Interview", "Storytelling", "Fast Facts", "Humble Brag", "Reflective"])  # noqa: E501
    parser.add_argument("--rebuild", action="store_true", help="Rebuild the index before querying")
    parser.add_argument("--reflective", action="store_true", help="Bias retrieval toward self_reflection.md")
    parser.add_argument("--temp", type=float, default=0.4)
    args = parser.parse_args()

    if args.rebuild:
        build_index()

    r, answer, sources = compose_answer(args.question, args.mode, args.reflective, args.temp)

    print("Retrieved sources:")
    for item in r["results"]:
        print(f"- {os.path.basename(item['source'])}#{item['heading']} (score={item['score']:.3f})")

    print("\nAnswer:\n")
    print(answer)
    print("\nSources:")
    print(sources)


if __name__ == "__main__":
    main()
