import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input

st.set_page_config(page_title="Movie Genres Classifier", page_icon="🎬")

st.title("Movie Poster Genre Classifier")
st.markdown(
    "Upload a movie poster and this app will predict which genres apply."
)

GENRE_NAMES = [
    "action", "adventure", "animation", "biography", "comedy",
    "crime", "documentary", "drama",  "family", "fantasy",
    "film-noir", "game-show", "history",  "horror",  "music",
    "musical",  "mystery", "news",  "reality-tv", "romance",
    "sci-fi", "sport", "talk-show",  "thriller", "war",
    "western"]


@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('my_model.keras', compile=False)
    return model

model = load_model()

def predict_genres(img_array, model, threshold=0.5):

    preds = model.predict(img_array)[0]  
    selected = [GENRE_NAMES[i] for i, p in enumerate(preds) if p >= threshold]
    if not selected:
        top_idx = np.argsort(preds)[-3:][::-1]
        selected = [GENRE_NAMES[i] for i in top_idx]
    return selected, preds

uploaded_file = st.file_uploader("Choose a poster image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.subheader("Uploaded Poster")
    st.image(uploaded_file, use_column_width=True)

    with st.spinner("Classifying..."):
        img = load_img(uploaded_file, target_size=(140, 207))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        genres, probs = predict_genres(img_array, model, threshold=0.5)

    st.subheader("Predicted Genres")
    st.markdown(
        f'<div style="border-left: 4px solid #4CAF50; padding: 8px 16px;">'
        f'<p style="font-size:18px;">{" , ".join(genres)}</p>'
        f'</div>',
        unsafe_allow_html=True
    )

    if st.checkbox("Show all probabilities"):
        import pandas as pd
        df = pd.DataFrame({
            "Genre": GENRE_NAMES,
            "Probability": np.round(probs, 3)
        }).sort_values("Probability", ascending=False)
        st.dataframe(df)
