import streamlit as st
from PIL import Image
from fastai.vision.all import *
from fastai.data.external import *
import os
import matplotlib.pyplot as plt

import streamlit as st


# Set page title and heading
st.title("Image Prediction App ⚙️")
st.header("Exploratory Data Analysis")

tab1, tab2, tab3 = st.tabs(["Step 1", "Step 2", "Step 3"])


with tab1:
    st.write("We counted the number of images per class")
    # display image "eda_count" in eda_images
    image_path = 'eda_images/eda_count.png'
    st.image(image_path, caption='Image Counts', use_column_width=True)
    st.write("We can see that the number of images per class is not balanced. We will have to take care of this later in the process.")


with tab2:
    st.write("We balanced the number of images per class")
    # display image "eda_count" in eda_images
    image_path = 'eda_images/balanced_dataset.png'
    st.image(image_path, caption='Image Counts', use_column_width=True)
    st.write(
        "We can see the number of images per class is now balanced and are now 400 each.")


with tab3:
    st.write(
        "We found the sizes of the images based on three disfferent categories. Small, Medium and Large")
    # display image "eda_count" in eda_images
    image_path = 'eda_images/sizes_display.png'
    st.image(image_path, caption='Image Counts', use_column_width=True)
    st.write(
        "We can see the number of images for the large files are a lot more than the small and medium files.")
    st.write()


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
# Load the FastAI learner model from the exported file
learner = load_learner('Own_model/second_model.pkl')


def is_allowed_file(filename):
    """
    Check if the file extension is allowed.

    Args:
        filename (str): The name of the file.

    Returns:
        bool: True if the file extension is allowed, False otherwise.
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


st.header("Model Inference")
# Upload a file
uploaded_file = st.file_uploader("Choose an image ...")

# Display heading only if an image is uploaded
if uploaded_file is not None:
    st.subheader("Uploaded Image:")
    if not is_allowed_file(uploaded_file.name):
        st.error(
            f"Invalid file type. Allowed types are {', '.join(ALLOWED_EXTENSIONS)}.")
    else:
        # Read the file
        image = Image.open(uploaded_file)
        # Display the image
        st.image(image, caption='Uploaded Image.', use_column_width=True)

        # Add a button for prediction
        if st.button('Predict'):
            # Make a prediction using the loaded learner model
            pred = learner.predict(image)
            # Display the prediction
            st.subheader("Prediction:")
            st.write(f"Prediction: {pred[0]}")
            st.write(f"Probability: {pred[2][pred[1]].item() * 100:.02f}%")

# Add some creativity
st.sidebar.header("About")
st.sidebar.write(
    "This is an Streamlit app that predicts the content of images. Upload an image and click the 'Predict' button to see the prediction.")
