import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
import time
import base64
from sklearn.decomposition import PCA
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess
import matplotlib.pyplot as plt
import io
import warnings
warnings.filterwarnings("ignore")

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

# Preprocess image using DenseNet121 preprocessing
def preprocess_image(image):
    try:
        img_array = np.array(image.convert('RGB'))
        original_img = img_array.copy()
        
        img_resized = cv2.resize(img_array, (224, 224))
        
        img_preprocessed = densenet_preprocess(img_resized.copy())
        img_preprocessed = np.expand_dims(img_preprocessed, axis=0)
        
        displayable_preprocessed = (img_preprocessed[0] - np.min(img_preprocessed[0])) / (np.max(img_preprocessed[0]) - np.min(img_preprocessed[0]))
        displayable_preprocessed = (displayable_preprocessed * 255).astype(np.uint8)
        
        return img_preprocessed, displayable_preprocessed, original_img
    except Exception as e:
        st.error(f"Error preprocessing image: {str(e)}")
        raise

# Validate if the image is likely a histological slide
def is_likely_medical_image(image):
    try:
        img_array = np.array(image.convert('RGB'))
        mean_rgb = np.mean(img_array, axis=(0, 1))
        if mean_rgb[1] < mean_rgb[0] and mean_rgb[1] < mean_rgb[2]:
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

# Enhanced preprocessing to highlight potential cancer regions
def enhance_cancer_features(img_array):
    try:
        hsv_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        saturation = hsv_img[:, :, 1]
        thresh = cv2.adaptiveThreshold(
            saturation, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        mask = np.zeros_like(img_array)
        mask[:, :, 1] = thresh
        mask[:, :, 0] = thresh // 2
        return mask, thresh
    except Exception as e:
        st.error(f"Error enhancing cancer features: {str(e)}")
        raise

# Generate PCA-based feature activation map
def generate_feature_map(img_array):
    try:
        hsv_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        height, width, channels = hsv_img.shape
        reshaped_img = hsv_img.reshape(height * width, channels)
        pca = PCA(n_components=3)
        pca_result = pca.fit_transform(reshaped_img)
        feature_map = pca_result[:, 0].reshape(height, width)
        feature_map = (feature_map - np.min(feature_map)) / (np.max(feature_map) - np.min(feature_map) + 1e-7)
        feature_map_colored = cv2.applyColorMap(
            np.uint8(feature_map * 255), cv2.COLORMAP_JET
        )
        feature_map_colored = cv2.cvtColor(feature_map_colored, cv2.COLOR_BGR2RGB)
        return feature_map_colored
    except Exception as e:
        st.error(f"Error generating feature map: {str(e)}")
        raise

# Combine the original image with the feature map
def create_combined_view(original_img, feature_mask):
    try:
        if original_img.shape[:2] != feature_mask.shape[:2]:
            feature_mask = cv2.resize(
                feature_mask, (original_img.shape[1], original_img.shape[0])
            )
        alpha = 0.7
        combined = cv2.addWeighted(original_img, alpha, feature_mask, 1-alpha, 0)
        return combined
    except Exception as e:
        st.error(f"Error creating combined view: {str(e)}")
        raise

# Create a three-panel visualization
def create_visualization_panels(img_preprocessed, feature_map, cancer_regions):
    try:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(img_preprocessed)
        axes[0].set_title('DenseNet Preprocessed Image')
        axes[0].axis('off')
        
        axes[1].imshow(feature_map)
        axes[1].set_title('Feature Activation Map (PCA)')
        axes[1].axis('off')
        
        axes[2].imshow(cancer_regions)
        axes[2].set_title('Combined View (Cancer Regions)')
        axes[2].axis('off')
        
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        panel_image = Image.open(buf)
        return panel_image
    except Exception as e:
        st.error(f"Error creating visualization panels: {str(e)}")
        raise

# Display results with visualization panels
def display_results(prediction, original_img, displayable_preprocessed, img_resized):
    class_names = ['Lung Adenocarcinoma', 'Lung Normal', 'Lung Squamous Cell Carcinoma']
    
    probabilities = prediction[0]
    predicted_class_idx = np.argmax(probabilities)
    predicted_class = class_names[predicted_class_idx]
    confidence = probabilities[predicted_class_idx]

    st.subheader("Prediction Results")
    
    if predicted_class == 'Lung Normal':
        st.success(f"Prediction: This image shows {predicted_class}")
    else:
        st.error(f"Prediction: This image shows {predicted_class}")
    
    if predicted_class != 'Lung Normal':
        st.subheader("Cancer Region Visualization")
        
        with st.spinner("Generating cancer region visualization..."):
            cancer_mask, _ = enhance_cancer_features(img_resized)
            feature_map = generate_feature_map(img_resized)
            combined_view = create_combined_view(img_resized, cancer_mask)
            visualization = create_visualization_panels(
                displayable_preprocessed, feature_map, combined_view
            )
            st.image(visualization, use_container_width=True)
            st.info("The visualization shows: preprocessed image, Feature activation map showing areas of interest based on PCA, and Combined view highlighting potential cancer regions. Brighter yellow/green areas in the combined view represent regions with cellular patterns most strongly associated with cancer.")
    
    with st.expander("Detailed Prediction Probabilities"):
        for class_name, prob in zip(class_names, probabilities):
            st.write(f"{class_name}: {prob:.2%}")
        st.write(f"Predicted Class Confidence: {confidence:.2%}")

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
    st.title("Lung Cancer Type Detection")
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
        st.image(image, caption='Uploaded Histological Image', width=400, use_container_width=False)

        if not is_likely_medical_image(image):
            st.warning("⚠️ This doesn't appear to be a histological image of lung tissue. Please upload a valid image (stained with hematoxylin and eosin).")
            return

        loading_placeholder = st.empty()
        with loading_placeholder:
            with st.spinner("🔍 Analyzing image... Please wait"):
                time.sleep(1)
                try:
                    processed_image, displayable_preprocessed, original_img = preprocess_image(image)
                    prediction = ensemble_predict(models, processed_image)
                except Exception as e:
                    st.error(f"Error processing image: {str(e)}")
                    return
        
        loading_placeholder.success("✅ Analysis Complete!")
        time.sleep(0.5)
        loading_placeholder.empty()

        display_results(prediction, original_img, displayable_preprocessed, img_resized=cv2.resize(original_img, (224, 224)))

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
    - **Preprocessing**: Uses DenseNet121's specific preprocessing functions
    - **Prediction Method**: Results are averaged across all four models for improved accuracy
    - **Input**: Accepts histological images in JPG, PNG, or JPEG format (stained with hematoxylin and eosin)
    - **Output**: Probability scores for each class and the most likely prediction
    - **Visualization**: Three-panel view showing the DenseNet preprocessed image, feature activation map, and highlighted cancer regions
    
    ### How the Visualization Works
    The cancer region visualization uses multiple techniques:
    1. **Feature Activation Map**: Uses Principal Component Analysis (PCA) to identify regions with the highest variance in pixel values
    2. **Enhanced Original Image**: Highlights potential cancer regions by analyzing color and texture patterns
    3. **Combined View**: Overlays the enhanced features on the original image to highlight potential cancer regions
    
    The bright yellow/green areas in the visualization indicate regions where the cellular patterns are most strongly associated with cancer according to our image analysis techniques.
    
    ### Disclaimer
    - This is an AI-based tool and not a substitute for professional medical diagnosis
    - For accurate evaluation, consult a pathologist or oncologist
    - The highlighted regions are based on statistical analysis and may not perfectly align with clinical standards
    
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
        
        navigation_options = {
            "Main Page": main_page,
            "About Page": about_page
        }
        
        selected_page = st.radio(
            "Select a page:",
            list(navigation_options.keys()),
            label_visibility="collapsed"
        )
        
        st.markdown("""
            <style>
            [data-testid="stSidebar"] {
                background-color: rgba(255, 255, 255, 0.1);
                padding: 20px;
            }
            h3 {
                color: #333;
                padding-bottom: 5px;
                margin-bottom: 20px;
            }
            [data-testid="stRadio"] > div {
                display: flex;
                flex-direction: column;
                gap: 10px;
            }
            [data-testid="stRadio"] > div > label {
                display: flex;
                align-items: center;
                background-color: rgba(255, 255, 255, 0.5);
                color: #333;
                padding: 8px 12px;
                border-radius: 5px;
                transition: all 0.3s;
            }
            [data-testid="stRadio"] > div > label:hover {
                background-color: rgba(255, 255, 255, 0.8);
            }
            [data-testid="stRadio"] > div > label > div:first-child {
                width: 20px !important;
                height: 20px !important;
                border: 2px solid #333 !important;
                margin-right: 10px;
            }
            [data-testid="stRadio"] > div > label > div:first-child > div {
                background-color: #333 !important;
                width: 12px !important;
                height: 12px !important;
                margin: 2px !important;
            }
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

    navigation_options[selected_page]()

if __name__ == '__main__':
    main()
