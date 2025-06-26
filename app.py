import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.xception import Xception, preprocess_input
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import json
from PIL import Image
import io

# Set Streamlit page config
st.set_page_config(page_title="Image Caption Generator", layout="centered")

st.title("🖼️ Image Caption Generator")
st.write("Upload an image and get an AI-generated caption using a trained deep learning model.")

# Load tokenizer and metadata
@st.cache_resource
def load_tokenizer_metadata():
    with open('tokenizer.json') as f:
        tokenizer = tokenizer_from_json(json.load(f))
    #with open('tokenizer.pkl', 'rb') as f:
        #tokenizer = pickle.load(f)
    with open('metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    return tokenizer, metadata['max_length']

tokenizer, max_length = load_tokenizer_metadata()

# Load trained model
@st.cache_resource
def load_caption_model():
    return load_model('best_model.keras')

model = load_caption_model()

# Load Xception model 
@st.cache_resource
def load_feature_extractor():
    return Xception(weights='imagenet', include_top=False, pooling='avg')

xception_model = load_feature_extractor()

# Convert integer to word
index_to_word = {i: w for w, i in tokenizer.word_index.items()}
def word_for_id(integer):
    return index_to_word.get(integer)

# Feature extractor
def extract_features(image):
    image = image.resize((299, 299)).convert('RGB')
    image_array = img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = preprocess_input(image_array)
    features = xception_model.predict(image_array, verbose=0)
    return features[0]

# Greedy decoding
def generate_caption_greedy(model, tokenizer, image_feature, max_length):
    in_text = 'start'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([np.expand_dims(image_feature, axis=0), sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = word_for_id(yhat)
        if word is None or word == 'end':
            break
        in_text += ' ' + word
    return ' '.join([w for w in in_text.split() if w not in ['start', 'end']])

# Beam search decoding
def beam_search(model, tokenizer, image_feature, max_length, beam_width=3):
    start_seq = [tokenizer.word_index['start']]
    sequences = [[start_seq, 0.0, ['start']]]
    image_feature_batch = np.expand_dims(image_feature, axis=0)

    for _ in range(max_length):
        all_candidates = []
        for seq, score, words in sequences:
            if words[-1] == 'end':
                all_candidates.append([seq, score, words])
                continue
            padded_seq = pad_sequences([seq], maxlen=max_length)
            yhat = model.predict([image_feature_batch, padded_seq], verbose=0)[0]
            top_k = np.argsort(yhat)[-beam_width:]
            for word_id in top_k:
                word = word_for_id(word_id)
                if word is None:
                    continue
                new_seq = seq + [word_id]
                new_words = words + [word]
                new_score = score + np.log(yhat[word_id] + 1e-10)
                all_candidates.append([new_seq, new_score, new_words])
        sequences = sorted(all_candidates, key=lambda tup: tup[1], reverse=True)[:beam_width]
        if all(seq[2][-1] == 'end' for seq in sequences):
            break

    final_caption = sequences[0][2]
    return ' '.join([w for w in final_caption if w not in ['start', 'end']])


# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Generating caption..."):
        features = extract_features(image)
        caption = beam_search(model, tokenizer, features, max_length, beam_width=3)

    st.markdown("### 📝 Caption:")
    st.success(caption)
