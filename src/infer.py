from openai import OpenAI

# vLLM ignores the api_key but the client requires one — any string works.
client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

MODEL = "granite-sft"  # matches --served-model-name in serve.sh

PROMPTS = [
    "Explain what overfitting is in one paragraph.",
    "Give me three tips for getting a daily drawing habit.",
    "Explain why 10+10 equals 20",
]


def main():
    for prompt in PROMPTS:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=256,
            temperature=0.7,
        )
        print("=" * 70)
        print("PROMPT:", prompt)
        print("-" * 70)
        print(resp.choices[0].message.content.strip())
        print()


if __name__ == "__main__":
    main()
