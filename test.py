import asyncio
import os
import time
import subprocess
from dotenv import load_dotenv
import edge_tts
from groq import Groq

load_dotenv()

VOICE = "en-GB-RyanNeural"
OUTPUT_FILE = "response.mp3"
PROMPT = "Give a short, friendly 1-sentence greeting for a voice assistant running on a Raspberry Pi."

async def generate_speech_from_text(text: str):
    """Generates an MP3 audio file from text using Edge-TTS."""
    print(f"\n[TTS] Generating audio with voice '{VOICE}'...")
    communicate = edge_tts.Communicate(
        text=text,
        voice=VOICE,
        rate="+0%",
        volume="+0%"
    )
    await communicate.save(OUTPUT_FILE)
    print(f"[TTS] Audio saved to {OUTPUT_FILE}")

def play_audio(filename: str):
    """Plays the generated MP3 file directly through Card 2 using mpg123."""
    print("[AUDIO] Playing response out loud...")
    subprocess.run([
        "mpg123",
        "-a", "sysdefault:CARD=2",
        filename
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("[AUDIO] Playback complete.")

async def main():
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    start_time = time.time()
    first_token_time = None
    full_response_text = ""

    print(f"--- Sending Prompt to Groq ---")
    print(f"Prompt: \"{PROMPT}\"")

    completion = client.chat.completions.create(
        model="qwen/qwen3.8-27b",
        messages=[{"role": "user", "content": PROMPT}],
        temperature=0.7,
        max_tokens=100,
        stream=True,
    )

    print("\n--- Response Stream ---")
    for chunk in completion:
        text = chunk.choices[0].delta.content or ""
        if text:
            if first_token_time is None:
                first_token_time = time.time() - start_time
                print(f"\n[TIME TO FIRST WORD: {first_token_time:.2f}s]\n")
            
            print(text, end="", flush=True)
            full_response_text += text

    print(f"\n\n[Total Generation Time: {time.time() - start_time:.2f}s]")

    if full_response_text:
        await generate_speech_from_text(full_response_text)
        
        play_audio(OUTPUT_FILE)

if __name__ == "__main__":
    asyncio.run(main())