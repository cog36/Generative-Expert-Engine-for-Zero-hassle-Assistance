import os
import time
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

start_time = time.time()
first_token_time = None

print("\n--- Sending Request to Groq ---")

# Pass max_tokens to stay well under Groq's free OTPM limit
completion = client.chat.completions.create(
    model="qwen/qwen3.8-27b",
    messages=[
        {"role": "user", "content": "Give a 1-sentence greeting for a new voice assistant running on a Raspberry Pi."}
    ],
    temperature=0.7,
    max_tokens=100,
    stream=True,
)

print("--- Response Stream ---")
for chunk in completion:
    text = chunk.choices[0].delta.content or ""
    if text and first_token_time is None:
        first_token_time = time.time() - start_time
        print(f"\n[TIME TO FIRST WORD: {first_token_time:.2f}s]\n")
    print(text, end="", flush=True)

print(f"\n\n[Total Generation Time: {time.time() - start_time:.2f}s]")