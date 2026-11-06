import os
import time
import asyncio
import subprocess
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
from scipy.signal import resample_poly
from dotenv import load_dotenv
from groq import Groq
import edge_tts
from openwakeword.model import Model

load_dotenv()

DEVICE_INDEX = 1
HW_SAMPLE_RATE = 48000
TARGET_SAMPLE_RATE = 16000
RECORD_DURATION = 5
WAV_FILENAME = "user_input.wav"
MP3_FILENAME = "response.mp3"

TTS_VOICE = "en-GB-RyanNeural"
LLM_MODEL = "qwen/qwen3.8-27b"

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
oww_model = Model()


def record_user_speech() -> str:
    print(f"\n[MIC] Listening... Speak now! ({RECORD_DURATION}s)")
    raw_audio = sd.rec(
        int(RECORD_DURATION * HW_SAMPLE_RATE),
        samplerate=HW_SAMPLE_RATE,
        channels=1,
        dtype='int16',
        device=DEVICE_INDEX
    )
    sd.wait()

    audio_16k = resample_poly(raw_audio.squeeze(), TARGET_SAMPLE_RATE, HW_SAMPLE_RATE).astype('int16')
    write(WAV_FILENAME, TARGET_SAMPLE_RATE, audio_16k)
    return WAV_FILENAME


def transcribe_audio(filename: str) -> str:
    print("[STT] Transcribing audio with Groq Whisper...")
    start_time = time.time()
    
    with open(filename, "rb") as file:
        transcription = groq_client.audio.transcriptions.create(
            file=(filename, file.read()),
            model="whisper-large-v3-turbo",
            response_format="json",
            language="en",
            temperature=0.0
        )
    
    elapsed = time.time() - start_time
    print(f"[STT] Result ({elapsed:.2f}s): \"{transcription.text}\"")
    return transcription.text.strip()


def get_llm_response(prompt_text: str) -> str:
    print(f"[LLM] Querying Groq ({LLM_MODEL})...")
    start_time = time.time()
    
    completion = groq_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": "You are a concise, helpful voice assistant named Geeza. Keep responses short and under 2 sentences."},
            {"role": "user", "content": prompt_text}
        ],
        temperature=0.7,
        max_tokens=100,
        stream=True,
    )

    full_text = ""
    for chunk in completion:
        text = chunk.choices[0].delta.content or ""
        if text:
            print(text, end="", flush=True)
            full_text += text
            
    print(f"\n[LLM] Response generated in {time.time() - start_time:.2f}s")
    return full_text


async def generate_tts(text: str):
    print(f"[TTS] Synthesizing speech ('{TTS_VOICE}')...")
    communicate = edge_tts.Communicate(text=text, voice=TTS_VOICE, rate="+0%", volume="+0%")
    await communicate.save(MP3_FILENAME)


def play_audio(filename: str):
    print("[AUDIO] Playing response...")
    subprocess.run([
        "mpg123",
        "-a", "sysdefault:CARD=2",
        filename
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("[AUDIO] Playback finished.")
    time.sleep(0.5)


def listen_for_wake_word():
    print("\n==================================================")
    print(" [GEEZA AI] System Ready! Say a wake word to begin.")
    print("==================================================")

    oww_model.reset()
    chunk_size = 3840
    
    with sd.InputStream(samplerate=HW_SAMPLE_RATE, channels=1, dtype='int16', device=DEVICE_INDEX) as stream:
        for _ in range(3):
            stream.read(chunk_size)
            
        while True:
            audio_chunk, _ = stream.read(chunk_size)
            raw_data = np.frombuffer(audio_chunk, dtype=np.int16)
            
            audio_16k = resample_poly(raw_data, TARGET_SAMPLE_RATE, HW_SAMPLE_RATE).astype('int16')
            
            prediction = oww_model.predict(audio_16k)

            for model_name, score in oww_model.prediction_buffer.items():
                if score[-1] > 0.4:
                    print(f"\n[WAKE WORD DETECTED] Triggered ({model_name}: {score[-1]:.2f})")
                    oww_model.reset()
                    return


async def main():
    while True:
        listen_for_wake_word()
        record_user_speech()
        user_text = transcribe_audio(WAV_FILENAME)

        if not user_text:
            print("[STT] No spoken text detected. Returning to listen mode...")
            oww_model.reset()
            continue

        ai_response = get_llm_response(user_text)

        if ai_response:
            await generate_tts(ai_response)
            play_audio(MP3_FILENAME)
            oww_model.reset()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[GEEZA AI] System shutting down safely...")