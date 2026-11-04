import asyncio
import edge_tts

TEXT = "Push it in my frontend"
VOICE = "en-GB-RyanNeural"
OUTPUT_FILE = "response.mp3"

async def generate_speech():
    print(f"Generating audio file with Edge-TTS using voice '{VOICE}'...")
    communicate = edge_tts.Communicate(
        text=TEXT, 
        voice=VOICE,
        rate="+0%",
        volume="+0%"
    )
    await communicate.save(OUTPUT_FILE)
    print(f"Audio saved successfully to {OUTPUT_FILE}")

if __name__ == "__main__":
    asyncio.run(generate_speech())