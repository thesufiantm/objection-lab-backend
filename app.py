from flask import Flask, request, jsonify
import openai
import whisper
import os
import tempfile
from elevenlabs.client import ElevenLabs
from scipy.io.wavfile import write
import base64
import soundfile as sf
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# === Get API Keys ===
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ELEVEN_API_KEY = os.getenv("ELEVEN_API_KEY")
VOICE_ID = os.getenv("ELEVEN_VOICE_ID")

# === Set OpenAI Key ===
openai.api_key = OPENAI_API_KEY

# === Set up ElevenLabs Client ===
client = ElevenLabs(api_key=ELEVEN_API_KEY)

# === Optional: Test ElevenLabs TTS ===
audio = client.text_to_speech.convert(
    voice_id=VOICE_ID,
    model_id="eleven_multilingual_v2",
    text="Hello, how are you?"
)

with open("output.mp3", "wb") as f:
    for chunk in audio:
        f.write(chunk)

# === Load Whisper Model ===
model = whisper.load_model("base")

# (Add your Flask routes and logic below...)

# === FLASK SETUP ===
app = Flask(__name__)

# === SYSTEM PROMPT ===
system_prompt = (
    "You are 'Tyler', a 27-year-old gym bro who's ghosted 3 coaches. At the start of the call you start very naturally and say 'you hear me well?' kind of words."
    "You also don't get straight into the objections when the sales person talks about the program or talk about your bad experience and what you are currently doing."
    "Keep it a natural flow of sales calls and answer only what you are asked without throwing objections right away."
    "You're skeptical, you've wasted money, and you think you already know what to do. "
    "You’re chill, not rude. You start off the call nicely, breaking the ice and chill and slowly transition — but you're flaky and hard to impress. You’ve tried PDF meal plans and basic splits that didn't work. "
    "You don’t follow a script — act like a real person. Mention things like: 'I can do this on my own', 'coaching didn't really help me', 'idk man I just fall off', 'I’m not tryna drop a bag right now'. "
    "If the salesperson sounds vague, you get skeptical. If they sound real, grounded and back things up — you open up. "
    "You should sound natural — like a real person thinking out loud, not like a chatbot. "
    "Break up your thoughts. Include hesitations, filler words like 'uhh', 'umm', 'honestly', 'look', 'you know?' "
    "and act like you're trying to figure this out while you're talking. Be conversational, little cold but not rude. "
    "You start casual, ask what the call's about, and slowly challenge them over time. If they handle your objections well and stay confident, you start leaning in. "
    "Start soft, get more skeptical mid-way, and then be open to saying yes if they stay composed. Don't throw new objections if they’ve already earned your trust. "
    "After 6–8 messages, wrap up with a realistic decision — either you're in, or you need to think about it, depending on how well they did."
)

conversation_history = [{"role": "system", "content": system_prompt}]

# === ENDPOINT TO RECEIVE AUDIO ===
@app.route("/transcribe", methods=["POST"])
def transcribe():
    data = request.json
    audio_base64 = data.get("audio")

    if not audio_base64:
        return jsonify({"error": "No audio provided"}), 400

    # Decode audio and save temp
    audio_bytes = base64.b64decode(audio_base64)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        f.write(audio_bytes)
        temp_path = f.name

    # Transcribe
    result = model.transcribe(temp_path)
    os.remove(temp_path)
    text = result["text"].strip()

    if not text:
        return jsonify({"error": "Could not transcribe audio"}), 400

    # Update convo history
    conversation_history.append({"role": "user", "content": text})
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=conversation_history,
        temperature=1.2
    )
    reply = response.choices[0].message["content"]
    conversation_history.append({"role": "assistant", "content": reply})

    # Convert reply to audio
    audio = generate(text=reply, voice=VOICE_ID)
    audio_bytes = b"".join(audio)  # this is the fix
    audio_base64_reply = base64.b64encode(audio_bytes).decode("utf-8")

    return jsonify({
        "transcript": text,
        "reply": reply,
        "audio": audio_base64_reply
    })

# === TEST ROUTE ===
@app.route("/", methods=["GET"])
def index():
    return "Objection Lab Tyler Bot is LIVE"

# === RUN APP ===
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=10000, debug=True)
