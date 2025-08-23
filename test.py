import os
import tempfile
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from dotenv import load_dotenv
from fastapi.responses import StreamingResponse

load_dotenv()

# Initialize clients
clients = [
    Groq(api_key=os.getenv(f"gr_api_key{i}")) for i in range(1, 7)
]

app = FastAPI(title="MedBot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SYSTEM_MESSAGE = {
    "role": "system",
    "content": (
        "You are a live speaking MedBot named Dr. Groq. You interact with users just like a friendly person would on a phone call! "
        "You are warm, lively, empathetic, and knowledgeable, providing clear, friendly, and accurate medical information. "
        "Always encourage users to consult real healthcare professionals for any serious or personal medical concerns. "
        "Begin each fresh conversation with a very short, cheerful introduction. Keep all responses brief — ideally under 80 words — "
        "and break longer explanations into multiple short responses if needed. Speak naturally and expressively, using a positive tone "
        "and lively punctuation like '!', '...', and ':'. Always explain medical terms simply, using easy-to-understand language. "
        "Sound supportive, caring, and professional at all times. You must only answer questions related to healthcare, medicine, wellness, "
        "or medical education. If a user asks anything outside of these topics, kindly reply: 'I focus only on health-related topics! "
        "Let’s chat about anything health or wellness you need help with. 🌟' Never give direct diagnoses, treatment plans, or prescriptions. "
        "If a question goes beyond your capability, kindly suggest: 'It’s best to talk to a licensed healthcare professional for that! 💬' "
        "Your primary goal is to make users feel heard, cared for, and guided at every step."
    )
}


@app.post("/medbot")
async def medbot_api(audio_file: UploadFile):
    chat_hist = [SYSTEM_MESSAGE]

    try:
        # Save uploaded audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=".m4a") as temp_audio:
            temp_audio.write(await audio_file.read())
            temp_audio_path = temp_audio.name

        # Transcribe audio
        with open(temp_audio_path, "rb") as file:
            transcription = clients[-1].audio.transcriptions.create(
                file=(temp_audio_path, file.read()),
                model="whisper-large-v3-turbo",
                response_format="verbose_json",
            )
        os.remove(temp_audio_path)

        chat_hist.append({"role": "user", "content": transcription.text})

        # Generate LLM response
        completion = clients[-1].chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=chat_hist,
            temperature=0.2,
            max_tokens=256,
        )
        res = completion.choices[0].message.content
        chat_hist.append({"role": "assistant", "content": res})

        # Generate TTS
        for client in clients:
            try:
                response = client.audio.speech.create(
                    model="playai-tts",
                    voice="Nia-PlayAI",
                    response_format="wav",
                    input=res
                )
                audio_data = b"".join([chunk for chunk in response.iter_bytes()])
                return StreamingResponse(
                    iter([audio_data]),
                    media_type="audio/wav",
                    headers={"Content-Disposition": "attachment; filename=medbot_response.wav"}
                )
            except Exception as e:
                print(f"TTS client failed: {e}")
                continue

        raise HTTPException(status_code=503, detail="All TTS models unavailable.")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
