import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from gtts import gTTS
import tempfile
import nltk

nltk.download('punkt')

# ---------------- LOAD BLIP ----------------

@st.cache_resource
def load_blip():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

processor, model = load_blip()

# ---------------- LOAD CAPTIONS ----------------

def load_captions(file_path):
    mapping = {}

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines[1:]:
        parts = line.strip().split(',', 1)
        if len(parts) < 2:
            continue

        img, caption = parts
        img = img.strip()
        caption = caption.strip().lower()

        if img not in mapping:
            mapping[img] = []

        mapping[img].append(caption)

    return mapping


@st.cache_data
def get_caption_mapping():
    return load_captions("captions.txt")

caption_mapping = get_caption_mapping()

# ---------------- BLIP ----------------

def blip_caption(image):
    inputs = processor(images=image, return_tensors="pt")
    out = model.generate(**inputs, max_length=30, num_beams=5)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

# ---------------- BLEU ----------------

def compute_bleu(references, candidate):
    smoothie = SmoothingFunction().method1
    references = [ref.split() for ref in references]
    candidate = candidate.split()
    return sentence_bleu(references, candidate, smoothing_function=smoothie)

# ---------------- TEXT TO SPEECH ----------------

def text_to_speech(text):
    tts = gTTS(text=text)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(temp_file.name)
    return temp_file.name

# ---------------- UI ----------------

st.title("🖼️ Voice Based Image Captioning (BLIP + BLEU Score)")

# 🔥 IMPORTANT NOTE
st.info("⚠️ NOTE: Upload images from Flickr30k dataset to get accurate BLEU score results.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img_name = uploaded_file.name

    st.image(image, caption=img_name, width='stretch')

    if st.button("Generate Caption"):

        caption = blip_caption(image)

        st.subheader("🤖 BLIP Caption:")
        st.write(caption)

        # 🔊 AUDIO OUTPUT
        audio_file = text_to_speech(caption)
        st.audio(audio_file)

        # ---------------- BLEU ----------------
        if img_name in caption_mapping:
            references = caption_mapping[img_name]
            bleu = compute_bleu(references, caption)

            st.subheader("📊 BLEU Score")
            st.write(f"BLIP: {bleu:.4f}")
        else:
            st.warning("No reference captions found. Please upload Flickr30k image.")