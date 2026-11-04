import os
import time
import sounddevice as sd
from scipy.io.wavfile import write
from scipy.signal import resample_poly
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# Configuration
HW_SAMPLE_RATE = 48000
TARGET_SAMPLE_RATE = 16000
DURATION = 5
FILENAME = "user_input.wav"
DEVICE_INDEX = 1

def record_audio():
    """Captures 48kHz native audio and resamples to 16kHz for Whisper."""
    print(f"\n[MIC] Recording for {DURATION} seconds at 48kHz... Speak now!")
    
    raw_audio = sd.rec(
        int(DURATION * HW_SAMPLE_RATE), 
        samplerate=HW_SAMPLE_RATE, 
        channels=1, 
        dtype='int16',
        device=DEVICE_INDEX
    )
    sd.wait()
    
    print("[MIC] Resampling audio 48kHz -> 16kHz...")
    audio_16k = resample_poly(raw_audio.squeeze(), TARGET_SAMPLE_RATE, HW_SAMPLE_RATE).astype('int16')
    
    write(FILENAME, TARGET_SAMPLE_RATE, audio_16k)
    print(f"[MIC] Audio processed and saved to {FILENAME}")

def transcribe_audio():
    """Sends the recorded WAV file to Groq Whisper for transcription."""
    print("[STT] Sending audio to Groq Whisper API...")
    start_time = time.time()
    
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    
    with open(FILENAME, "rb") as file:
        transcription = client.audio.transcriptions.create(
            file=(FILENAME, file.read()),
            model="whisper-large-v3-turbo",
            response_format="json",
            language="en",
            temperature=0.0
        )
    
    elapsed_time = time.time() - start_time
    print(f"\n[STT RESULT] \"{transcription.text}\"")
    print(f"[STT TIME] Transcribed in {elapsed_time:.2f}s")

if __name__ == "__main__":
    record_audio()
    transcribe_audio()