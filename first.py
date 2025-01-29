import librosa
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# Load model and processor locally
device = "cuda" if torch.cuda.is_available() else "cpu"

try:
    print("🔄 Loading model from 'whisper_model'...")
    model_name = "openai/whisper-medium"
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)
    print("✅ Model loaded successfully!")
except Exception as e:
    print("❌ Error loading model:", e)
    exit(1)  # Exit script if model fails to load


# Load model and processor locally
device = "cuda" if torch.cuda.is_available() else "cpu"


def transcribe(audio_path):
    print(f"📢 Processing file: {audio_path}")

    # Load audio
    audio, sr = librosa.load(audio_path, sr=16000)
    print(f"✅ Loaded {len(audio)} samples at {sr} Hz")

    if len(audio) == 0:
        print("❌ Audio is empty!")
        return "Error: Empty Audio"

    # Tokenize input
    input_features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features.to(device)
    print(f"🔄 Input feature shape: {input_features.shape}")

    if input_features.shape[1] == 0:
        print("❌ Input features are empty! Check audio file.")
        return "Error: Empty Input Features"

    # Generate transcription
    try:
        print("🎙️ Generating transcription...")
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
        input_features = inputs.input_features.to(device)
        attention_mask = inputs.attention_mask.to(device)

        predicted_ids = model.generate(input_features, attention_mask=attention_mask)
        # print(f"🔤 Predicted token IDs Shape: {predicted_ids.shape}")
        # print(f"🔤 Predicted token IDs: {predicted_ids}")
    except Exception as e:
        print("❌ Error during generation:", e)
        return "Error in Model Generation"

    # Check if output is empty
    if predicted_ids.numel() == 0:
        print("❌ Model returned empty output!")
        return "No transcription"

    # Decode transcription
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription


# Test Function
text = transcribe("rec2.wav")
print("📝 Transcription:", text)
