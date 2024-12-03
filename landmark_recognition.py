import streamlit as st
import PIL
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd
from geopy.geocoders import Nominatim

# URL for the model hosted on TensorFlow Hub
model_url = 'https://tfhub.dev/google/on_device_vision/classifier/landmarks_classifier_asia_V1/1'

# Load the labels from the CSV file
labels = 'landmarks_classifier_asia_V1_label_map.csv'
df = pd.read_csv(labels)
labels = dict(zip(df.id, df.name))

# Image processing function to resize, normalize, and classify the uploaded image
def image_processing(image):
    img_shape = (321, 321)
    
    # Load the TensorFlow Hub model directly
    classifier = hub.KerasLayer(model_url, input_shape=img_shape + (3,), output_key="predictions:logits")
    
    # Open and resize the image
    img = PIL.Image.open(image)
    img = img.resize(img_shape)
    img1 = img
    img = np.array(img) / 255.0  # Normalize the image
    img = img[np.newaxis]  # Add batch dimension

    # Get the prediction from the model
    result = classifier(img)  # Calling the model directly to get predictions
    
    # Return the predicted label and the processed image
    return labels[np.argmax(result)], img1

# Function to get the address and coordinates of a location using geopy
def get_map(loc):
    geolocator = Nominatim(user_agent="Your_Name")
    location = geolocator.geocode(loc)
    return location.address, location.latitude, location.longitude

# Main function to run the Streamlit app
def run():
    st.title("Landmark Detection")
    
    # Display the logo
    img = PIL.Image.open('logo.jpeg')
    img = img.resize((256, 256))
    st.image(img)
    
    # File uploader to choose an image
    img_file = st.file_uploader("Choose your Image", type=['png', 'jpg'])
    
    if img_file is not None:
        # Save the uploaded image
        save_image_path = './Uploaded_Images/' + img_file.name
        with open(save_image_path, "wb") as f:
            f.write(img_file.getbuffer())
        
        # Process the image and get predictions
        prediction, image = image_processing(save_image_path)
        
        # Display the image and the predicted label
        st.image(image)
        st.header(f"📍 **Predicted Landmark is: {prediction}**")
        
        # Try to get the address and coordinates of the landmark
        try:
            address, latitude, longitude = get_map(prediction)
            st.success(f'Address: {address}')
            
            loc_dict = {'Latitude': latitude, 'Longitude': longitude}
            st.subheader(f'✅ **Latitude & Longitude of {prediction}**')
            st.json(loc_dict)
            
            # Display the landmark location on the map
            data = [[latitude, longitude]]
            df = pd.DataFrame(data, columns=['lat', 'lon'])
            st.subheader(f'✅ **{prediction} on the Map** 🗺️')
            st.map(df)
        
        except Exception as e:
            st.warning("No address found!!")

# Run the Streamlit app
run()
