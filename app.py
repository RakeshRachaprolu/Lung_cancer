import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
import time
import base64

# Set page config with icon (must be at the top)
st.set_page_config(
    page_title="Lung Cancer Detection",
    page_icon="🫁",
    layout="wide"
)

# Load four pre-trained models
@st.cache_resource
def load_models():
    try:
        model1 = tf.keras.models.load_model('ensemble_model/densenet121.h5')
        model2 = tf.keras.models.load_model('ensemble_model/inceptionv3.h5')
        model3 = tf.keras.models.load_model('ensemble_model/mobilenet.h5')
        model4 = tf.keras.models.load_model('ensemble_model/vgg16.h5')
        return model1, model2, model3, model4
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        raise

# Preprocess image
def preprocess_image(image):
    try:
        img_array = np.array(image.convert('RGB'))
        img_array = cv2.resize(img_array, (224, 224))
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        st.error(f"Error preprocessing image: {str(e)}")
        raise

# Validate if the image is likely a histological slide
def is_likely_medical_image(image):
    try:
        img_array = np.array(image.convert('RGB'))
        # Check for characteristics of histological images
        # Histological images often have a mix of purple (hematoxylin) and pink (eosin) colors
        # We can check for the presence of these colors by looking at the RGB channels
        mean_rgb = np.mean(img_array, axis=(0, 1))
        # Typical histological images have a purple/pink hue, so blue (B) and red (R) channels dominate
        # Green (G) is usually lower due to the staining
        if mean_rgb[1] < mean_rgb[0] and mean_rgb[1] < mean_rgb[2]:  # Green channel is lower
            return True
        return False
    except Exception as e:
        st.error(f"Error validating image: {str(e)}")
        return False

# Ensemble prediction with four models
def ensemble_predict(models, input_data):
    try:
        model1, model2, model3, model4 = models
        pred1 = model1.predict(input_data)
        pred2 = model2.predict(input_data)
        pred3 = model3.predict(input_data)
        pred4 = model4.predict(input_data)
        ensemble_pred = (pred1 + pred2 + pred3 + pred4) / 4
        return ensemble_pred
    except Exception as e:
        st.error(f"Error in ensemble prediction: {str(e)}")
        raise

# Display results with confidence threshold
def display_results(prediction):
    class_names = ['Lung Adenocarcinoma', 'Lung Normal', 'Lung Squamous Cell Carcinoma']
    probabilities = prediction[0]
    predicted_class_idx = np.argmax(probabilities)
    predicted_class = class_names[predicted_class_idx]
    confidence = probabilities[predicted_class_idx]

    st.subheader("Prediction Results")
    # if confidence < 0.7:  # Adjust threshold as needed
    #     st.warning("⚠️ Low confidence prediction. This may not be a valid histological image.")
    # else:
    if predicted_class == 'Lung Normal':
            st.success(f"Prediction: This image shows {predicted_class}")
    else:
            st.error(f"Prediction: This image shows {predicted_class}")
        
    # st.write("Probabilities:")
    # for class_name, prob in zip(class_names, probabilities):
    #         st.write(f"{class_name}: {prob:.2%}")
        
    # st.write(f"Predicted Class Confidence: {confidence:.2%}")

# Function to convert local image to base64
def get_base64_image(file_path):
    try:
        with open(file_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except Exception as e:
        st.error(f"Error loading background image: {str(e)}")
        return None

# Add background image and style navigation bar
def set_background_and_styles():
    image_path = "bg1.jpg"
    bg_image = get_base64_image(image_path)
    
    if bg_image:
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url(data:image/jpg;base64,{bg_image});
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
                background-attachment: fixed;
            }}
            /* Style main content for readability */
            .main .block-container {{
                background-color: rgba(255, 255, 255, 0.9);
                padding: 2rem;
                border-radius: 10px;
                max-width: 1200px;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )

# Main page
def main_page():
    st.title("Lung Cancer Type Detection ")
    st.write("Upload a histological image of lung tissue to classify as Normal, Adenocarcinoma, or Squamous Cell Carcinoma")
    
    with st.spinner("Loading models..."):
        models = load_models()

    uploaded_file = st.file_uploader(
        "Upload Histological Image",
        type=['jpg', 'png', 'jpeg'],
        help="Upload a clear histological image of lung tissue (stained with hematoxylin and eosin)."
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Histological Image', width=400)

        # Validate if the image is likely a histological slide
        if not is_likely_medical_image(image):
            st.warning("⚠️ This doesn't appear to be a histological image of lung tissue. Please upload a valid image (stained with hematoxylin and eosin).")
            return

        loading_placeholder = st.empty()
        with loading_placeholder:
            with st.spinner("🔍 Analyzing image... Please wait"):
                time.sleep(1)
                try:
                    processed_image = preprocess_image(image)
                    prediction = ensemble_predict(models, processed_image)
                except Exception as e:
                    st.error(f"Error processing image: {str(e)}")
                    return
        
        loading_placeholder.success("✅ Analysis Complete!")
        time.sleep(0.5)
        loading_placeholder.empty()

        display_results(prediction)

# About page
def about_page():
    st.title("About Lung Cancer Type Detection")
    st.write("""
    ### Overview
    This application uses deep learning to classify histological images of lung tissue into three categories:
    - Lung Normal
    - Lung Adenocarcinoma
    - Lung Squamous Cell Carcinoma
    
    ### Technology
    - **Ensemble of 4 Models**: DenseNet121, InceptionV3, MobileNet, and VGG16
    - **Prediction Method**: Results are averaged across all four models for improved accuracy
    - **Input**: Accepts histological images in JPG, PNG, or JPEG format (stained with hematoxylin and eosin)
    - **Output**: Probability scores for each class and the most likely prediction
    
    ### Disclaimer
    - This is an AI-based tool and not a substitute for professional medical diagnosis
    - For accurate evaluation, consult a pathologist or oncologist
    
    ### Contact
    For questions or feedback, please reach out to the development team.
    """)

# Main function with improved radio button navigation
def main():
    set_background_and_styles()
    st.header("Lung Cancer Detection App")

    with st.sidebar:
        st.markdown("""
            <div style='text-align: center; margin-bottom: 20px;'>
                <span style='font-size: 40px;'>🫁</span>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### Menu")
        
        # Radio button navigation
        navigation_options = {
            "Main Page": main_page,
            "About Page": about_page
        }
        
        selected_page = st.radio(
            "Select a page:",
            list(navigation_options.keys()),
            label_visibility="collapsed"  # Hide the default label
        )
        
        # Custom styling for radio buttons
        st.markdown("""
            <style>
            /* Style sidebar container */
            [data-testid="stSidebar"] {
                background-color: rgba(255, 255, 255, 0.1);
                padding: 20px;
            }
            /* Style sidebar title (removed border-bottom) */
            h3 {
                color: #333;
                padding-bottom: 5px;
                margin-bottom: 20px;
            }
            /* Style radio buttons container */
            [data-testid="stRadio"] > div {
                display: flex;
                flex-direction: column;
                gap: 10px;
            }
            /* Style each radio button label */
            [data-testid="stRadio"] > div > label {
                display: flex;
                align-items: center;
                background-color: rgba(255, 255, 255, 0.5);
                color: #333;
                padding: 8px 12px;
                border-radius: 5px;
                transition: all 0.3s;
            }
            /* Hover effect */
            [data-testid="stRadio"] > div > label:hover {
                background-color: rgba(255, 255, 255, 0.8);
            }
            /* Style the radio button circle */
            [data-testid="stRadio"] > div > label > div:first-child {
                width: 20px !important;
                height: 20px !important;
                border: 2px solid #333 !important;
                margin-right: 10px;
            }
            /* Style the selected radio button circle */
            [data-testid="stRadio"] > div > label > div:first-child > div {
                background-color: #333 !important;
                width: 12px !important;
                height: 12px !important;
                margin: 2px !important;
            }
            /* Remove default Streamlit radio button border */
            [data-testid="stRadio"] > div > label > div:first-child > div[aria-checked="true"] {
                border: none !important;
            }
            </style>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("""
            <div style='font-size: 12px; color: #666;'>
                Version 1.0.0<br>
                Last Updated: March 2025
            </div>
        """, unsafe_allow_html=True)

    # Display the selected page
    navigation_options[selected_page]()

if __name__ == '__main__':
    main()
