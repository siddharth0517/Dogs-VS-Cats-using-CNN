import streamlit as st
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from PIL import Image

# Load the pre-trained CNN model
model = load_model('cat_dog_classifier.h5')

# Streamlit app title and description
st.title('Cat and Dog Classification using CNN')
st.write('Upload an image of a cat or a dog, and the model will predict the class.')

# File uploader for image input
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess the image
    def preprocess_image(img):
        img = img.resize((64, 64))  # Resize to match the CNN input size
        img = img.convert('RGB')  # Ensure image has 3 channels
        img = np.array(img)  # Convert to numpy array
        img = img / 255.0  # Scale pixel values
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        return img

    # Convert the file to an image and preprocess it
    img = Image.open(uploaded_file)
    processed_img = preprocess_image(img)
    
    # Make prediction
    prediction = model.predict(processed_img)
    
    # Display the result
    if prediction[0][0] > 0.5:
        st.write("It's a **Dog**!")
    else:
        st.write("It's a **Cat**!")
