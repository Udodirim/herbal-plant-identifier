import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os
from pathlib import Path
import plotly.graph_objects as go

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="🌿 Herbal Plant Identifier",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM STYLING
# ============================================================================

st.markdown("""
<style>
    .main-title {
        text-align: center;
        color: #2d5016;
        margin-bottom: 30px;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .confidence-high {
        color: #28a745;
        font-weight: bold;
    }
    .confidence-medium {
        color: #ffc107;
        font-weight: bold;
    }
    .confidence-low {
        color: #dc3545;
        font-weight: bold;
    }
    .stTabs [data-baseweb="tab-list"] button {
        font-size: 16px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD MODEL AND METADATA
# ============================================================================

@st.cache_resource
def load_model_and_metadata():
    """Load trained model and metadata"""
    try:
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, "herbal_plant_model", "herbal_plant_model.h5")
        metadata_path = os.path.join(script_dir, "herbal_plant_model", "model_metadata.json")
        
        # Load model
        model = tf.keras.models.load_model(model_path)
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Create idx_to_class mapping
        idx_to_class = {int(k): v for k, v in metadata['idx_to_class'].items()}
        class_to_idx = metadata['class_mapping']
        
        return model, metadata, idx_to_class, class_to_idx
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.info("Make sure the model files are in `herbal_plant_model/` folder")
        return None, None, None, None

# ============================================================================
# PREDICTION FUNCTION
# ============================================================================

def predict_plant(image_array, model, idx_to_class):
    """Make prediction on an image"""
    try:
        # Resize to model input size
        img_resized = tf.image.resize(image_array, (224, 224))
        
        # Normalize
        img_normalized = img_resized / 255.0
        
        # Add batch dimension
        img_batch = np.expand_dims(img_normalized, axis=0)
        
        # Predict
        predictions = model.predict(img_batch, verbose=0)
        pred_probabilities = predictions[0]
        
        # Get top predictions
        top_3_idx = np.argsort(pred_probabilities)[-3:][::-1]
        top_predictions = [
            {
                'plant': idx_to_class.get(int(idx), idx_to_class.get(str(idx), f"Unknown (idx: {idx})")),
                'confidence': float(pred_probabilities[idx]),
                'percentage': float(pred_probabilities[idx] * 100)
            }
            for idx in top_3_idx
        ]
        
        return top_predictions
    except Exception as e:
        st.error(f"❌ Prediction error: {e}")
        return None

# ============================================================================
# CONFIDENCE COLOR FUNCTION
# ============================================================================

def get_confidence_color(confidence):
    """Get color based on confidence level"""
    if confidence >= 0.75:
        return "confidence-high"
    elif confidence >= 0.5:
        return "confidence-medium"
    else:
        return "confidence-low"

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Load model
    model, metadata, idx_to_class, class_to_idx = load_model_and_metadata()
    
    if model is None:
        st.error("⚠️ Cannot proceed without model. Please check the model files.")
        return
    
    # Header
    st.markdown("<h1 class='main-title'>🌿 Herbal Plant Identifier</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: gray;'>Powered by VGG16 CNN • 37 Plant Classes</p>", 
                unsafe_allow_html=True)
    
    # Sidebar - Info
    with st.sidebar:
        st.header("🌱 Available Plants")
        plant_list = sorted(class_to_idx.keys())
        with st.expander("View all plant classes"):
            for i, plant in enumerate(plant_list, 1):
                st.caption(f"{i}. {plant}")
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["📤 Upload Image", "📸 Camera Capture", "📊 Batch Analysis"])
    
    # ========================================================================
    # TAB 1: UPLOAD IMAGE
    # ========================================================================
    with tab1:
        st.subheader("Upload a Plant Image")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            uploaded_file = st.file_uploader(
                "Choose an image...",
                type=["jpg", "jpeg", "png", "bmp", "gif"],
                help="Upload a clear image of a plant"
            )
            
            if uploaded_file is not None:
                # Display uploaded image
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_container_width=True)
                
                # Convert to numpy array
                image_array = np.array(image.convert('RGB'), dtype=np.float32)
                
                # Make prediction
                if st.button("🔍 Identify Plant", key="predict_upload", use_container_width=True):
                    with st.spinner("Analyzing image..."):
                        st.session_state.predictions_upload = predict_plant(image_array, model, idx_to_class)
                    
                    if st.session_state.predictions_upload:
                        st.success("✅ Prediction Complete!")
                        st.rerun()
        
        with col2:
            if uploaded_file is not None and 'predictions_upload' in st.session_state and st.session_state.predictions_upload:
                st.subheader("🎯 Prediction Results")
                
                # Top prediction
                predictions = st.session_state.predictions_upload
                top_pred = predictions[0]
                confidence = top_pred['confidence']
                
                st.markdown(f"""
                <div class='prediction-box'>
                    <h3 style='margin-top: 0;'>Predicted Plant</h3>
                    <h2 style='color: white;'>{top_pred['plant']}</h2>
                    <p style='font-size: 18px;'>
                        Confidence: <span class='{get_confidence_color(confidence)}'>{top_pred['percentage']:.1f}%</span>
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Confidence gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=confidence * 100,
                    title={'text': "Confidence Level"},
                    delta={'reference': 80},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgray"},
                            {'range': [50, 75], 'color': "lightyellow"},
                            {'range': [75, 100], 'color': "lightgreen"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ))
                fig.update_layout(height=350, margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(fig, use_container_width=True)
    
    # ========================================================================
    # TAB 2: CAMERA CAPTURE
    # ========================================================================
    with tab2:
        st.subheader("📸 Capture Plant Image with Camera")
        st.info("📱 Click the camera icon to take a photo of the plant")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            camera_image = st.camera_input("Take a picture")
            
            if camera_image is not None:
                image = Image.open(camera_image)
                st.image(image, caption="Captured Image", use_container_width=True)
                
                image_array = np.array(image.convert('RGB'), dtype=np.float32)
                
                if st.button("🔍 Identify Plant", key="predict_camera", use_container_width=True):
                    with st.spinner("Analyzing image..."):
                        st.session_state.predictions_camera = predict_plant(image_array, model, idx_to_class)
                    
                    if st.session_state.predictions_camera:
                        st.success("✅ Prediction Complete!")
                        st.rerun()
        
        with col2:
            if camera_image is not None and 'predictions_camera' in st.session_state and st.session_state.predictions_camera:
                st.subheader("🎯 Prediction Results")
                
                predictions = st.session_state.predictions_camera
                top_pred = predictions[0]
                confidence = top_pred['confidence']
                
                st.markdown(f"""
                <div class='prediction-box'>
                    <h3 style='margin-top: 0;'>Predicted Plant</h3>
                    <h2 style='color: white;'>{top_pred['plant']}</h2>
                    <p style='font-size: 18px;'>
                        Confidence: <span class='{get_confidence_color(confidence)}'>{top_pred['percentage']:.1f}%</span>
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Confidence gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=confidence * 100,
                    title={'text': "Confidence Level"},
                    delta={'reference': 80},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgray"},
                            {'range': [50, 75], 'color': "lightyellow"},
                            {'range': [75, 100], 'color': "lightgreen"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ))
                fig.update_layout(height=350, margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(fig, use_container_width=True)
    
    # ========================================================================
    # TAB 3: BATCH ANALYSIS
    # ========================================================================
    with tab3:
        st.subheader("📊 Batch Image Analysis")
        st.info("Upload multiple images to analyze them all at once")
        
        uploaded_files = st.file_uploader(
            "Choose multiple images...",
            type=["jpg", "jpeg", "png", "bmp", "gif"],
            accept_multiple_files=True,
            help="Upload multiple plant images"
        )
        
        if uploaded_files:
            st.write(f"📁 {len(uploaded_files)} image(s) selected")
            
            if st.button("🔍 Analyze All Images", use_container_width=True):
                results = []
                progress_bar = st.progress(0)
                
                for idx, file in enumerate(uploaded_files):
                    image = Image.open(file)
                    image_array = np.array(image.convert('RGB'), dtype=np.float32)
                    predictions = predict_plant(image_array, model, idx_to_class)
                    
                    if predictions:
                        results.append({
                            'file': file.name,
                            'image': image,
                            'predictions': predictions
                        })
                    
                    progress_bar.progress((idx + 1) / len(uploaded_files))
                
                st.success(f"✅ Analyzed {len(results)}/{len(uploaded_files)} images")
                
                # Display results
                st.subheader("📋 Results Summary")
                
                for result in results:
                    with st.expander(f"📄 {result['file']}", expanded=False):
                        col1, col2 = st.columns([1, 1.5])
                        
                        with col1:
                            st.image(result['image'], caption=result['file'], use_container_width=True)
                        
                        with col2:
                            top_pred = result['predictions'][0]
                            confidence = top_pred['confidence']
                            
                            st.markdown(f"""
                            <div class='prediction-box'>
                                <h4 style='margin-top: 0;'>Top Prediction</h4>
                                <h3 style='color: white;'>{top_pred['plant']}</h3>
                                <p style='font-size: 16px;'>
                                    Confidence: <span class='{get_confidence_color(confidence)}'>{top_pred['percentage']:.1f}%</span>
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.write("**All Predictions:**")
                            for i, pred in enumerate(result['predictions'], 1):
                                st.write(f"{i}. {pred['plant']}: **{pred['percentage']:.1f}%**")
    
    # Footer
    st.divider()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.caption("🌿 Herbal Plant Identifier v1.0")
    with col2:
        st.caption("✨ Powered by VGG16 CNN")
    with col3:
        st.caption("📊 37 Plant Classes")

if __name__ == "__main__":
    main()
