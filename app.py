import streamlit as st
import cv2
import numpy as np
import os
import pickle
from keras.layers import GlobalMaxPooling2D
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras import Sequential
from PIL import Image
from sklearn.neighbors import NearestNeighbors

# Model initialization
conv_layer = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
conv_layer.trainable = False

model = Sequential()
model.add(conv_layer)
model.add(GlobalMaxPooling2D())

# Load precomputed feature embeddings
features_array = pickle.load(open('features_array.pkl', 'rb'))

# Initialize Nearest Neighbors
neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
neighbors.fit(features_array)

# Function to extract image features
def extract_image_features(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    image_array = np.array(image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = preprocess_input(image_array)
    features = model.predict(image_array)
    return features / np.linalg.norm(features)

# Function to get recommendations
def get_recommendation(image_path):
    image_features = extract_image_features(image_path)
    distances, indices = neighbors.kneighbors(image_features)
    return distances, indices[0]

# Streamlit UI setup
st.title('Fashion Recommender System')

uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    # Save the uploaded image temporarily
    image_path = os.path.join("images", uploaded_file.name)
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Display the uploaded image
    uploaded_image = Image.open(uploaded_file)
    st.image(uploaded_image, caption="Uploaded Image", width=150)

    # Get recommendations based on the uploaded image
    distances, indices = get_recommendation(image_path)
    
    # Search dynamically in the 'images' directory
    st.subheader("Recommended Images:")
    image_files = sorted(os.listdir("images"))  # Sort to maintain order consistency
    for i in indices[1:]:
        recommended_image_path = os.path.join("images", image_files[i])
        recommended_image = Image.open(recommended_image_path)
        st.image(recommended_image, caption=f"Recommendation: {image_files[i]}", width=150)
