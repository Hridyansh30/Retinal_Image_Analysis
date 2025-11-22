import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import (Layer, Dense, Conv2D, Multiply, Input, 
                                     UpSampling2D, concatenate, BatchNormalization, 
                                     Activation)
import numpy as np
from PIL import Image
import io

from huggingface_hub import hf_hub_download


# Set page config
st.set_page_config(
    page_title="Fundus Image Analysis",
    page_icon="👁️",
    layout="wide"
)

# Disease classes
DISEASE_CLASSES = ['amd', 'diabetic_retinopathy', 'glaucoma', 'normal']

# Define CBAM layer (for segmentation model)
class CBAM(Layer):
    def __init__(self, reduction_ratio=8, **kwargs):
        super(CBAM, self).__init__(**kwargs)
        self.reduction_ratio = reduction_ratio
        self.shared_dense_layers = {}

    def build(self, input_shape):
        channel = input_shape[-1]
        if channel not in self.shared_dense_layers:
            shared_dense_one = Dense(channel // self.reduction_ratio, activation='relu')
            shared_dense_two = Dense(channel)
            self.shared_dense_layers[channel] = (shared_dense_one, shared_dense_two)
        self.conv_spatial = Conv2D(1, kernel_size=7, padding='same', activation='sigmoid')

    def call(self, x):
        channel = x.shape[-1]
        shared_dense_one, shared_dense_two = self.shared_dense_layers[channel]
        
        # Channel Attention
        avg_pool = tf.reduce_mean(x, axis=[1, 2])
        max_pool = tf.reduce_max(x, axis=[1, 2])
        avg_out = shared_dense_two(shared_dense_one(avg_pool))
        max_out = shared_dense_two(shared_dense_one(max_pool))
        channel_attention = tf.nn.sigmoid(avg_out + max_out)
        channel_attention = tf.reshape(channel_attention, [-1, 1, 1, channel])
        x = Multiply()([x, channel_attention])
        
        # Spatial Attention
        avg_pool = tf.reduce_mean(x, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(x, axis=-1, keepdims=True)
        concat = tf.concat([avg_pool, max_pool], axis=-1)
        spatial_attention = self.conv_spatial(concat)
        x = Multiply()([x, spatial_attention])
        return x

    def get_config(self):
        config = super(CBAM, self).get_config()
        config.update({"reduction_ratio": self.reduction_ratio})
        return config

# Segmentation metrics and loss functions
def dice_metric(y_true, y_pred, smooth=1):
    y_pred_bin = tf.cast(y_pred > 0.5, tf.float32)
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred_bin)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred, smooth=1):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

def weighted_bce(y_true, y_pred, weight=5):
    bce = tf.keras.backend.binary_crossentropy(y_true, y_pred)
    return tf.reduce_mean(tf.where(tf.equal(y_true, 1), bce * weight, bce))

def combined_loss_weighted(y_true, y_pred):
    return dice_loss(y_true, y_pred) + weighted_bce(y_true, y_pred)

# Build segmentation model architecture
def conv_bn_relu(x, filters, kernel_size=3, padding="same", strides=1):
    x = Conv2D(filters, kernel_size, padding=padding, strides=strides)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(filters, kernel_size, padding=padding, strides=1)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x

def build_unet_cbam(input_shape=(256, 256, 3)):
    inputs = Input(shape=input_shape)
    base_model = EfficientNetB0(weights=None, include_top=False, input_tensor=inputs)
    
    # Encoder features
    C1 = base_model.get_layer("stem_activation").output
    C2 = base_model.get_layer("block2a_activation").output
    C3 = base_model.get_layer("block3a_activation").output
    C4 = base_model.get_layer("block4a_activation").output
    C5 = base_model.get_layer("top_activation").output
    
    # Decoder
    up1 = UpSampling2D()(C5)
    concat1 = concatenate([up1, C4])
    conv1 = conv_bn_relu(concat1, 512)
    conv1 = CBAM()(conv1)
    
    up2 = UpSampling2D()(conv1)
    concat2 = concatenate([up2, C3])
    conv2 = conv_bn_relu(concat2, 256)
    conv2 = CBAM()(conv2)
    
    up3 = UpSampling2D()(conv2)
    concat3 = concatenate([up3, C2])
    conv3 = conv_bn_relu(concat3, 128)
    conv3 = CBAM()(conv3)
    
    up4 = UpSampling2D()(conv3)
    concat4 = concatenate([up4, C1])
    conv4 = conv_bn_relu(concat4, 64)
    conv4 = CBAM()(conv4)
    
    up5 = UpSampling2D()(conv4)
    conv5 = conv_bn_relu(up5, 32)
    conv5 = CBAM()(conv5)
    
    output = Conv2D(1, (1, 1), activation="sigmoid")(conv5)
    
    model = Model(inputs, output)
    return model

# Load models with caching
@st.cache_resource
def load_segmentation_model(model_path, load_method='direct'):
    """Load the trained segmentation model"""
    custom_objects = {
        'CBAM': CBAM,
        'dice_metric': dice_metric,
        'dice_loss': dice_loss,
        'weighted_bce': weighted_bce,
        'combined_loss_weighted': combined_loss_weighted
    }
    
    if load_method == 'direct':
        try:
            model = load_model(model_path, custom_objects=custom_objects, compile=False)
            model.compile(
                optimizer='adam',
                loss=combined_loss_weighted,
                metrics=['accuracy', dice_metric]
            )
            return model
        except Exception as e:
            st.warning(f"Direct loading failed: {str(e)}")
            load_method = 'weights'
    
    if load_method == 'weights':
        model = build_unet_cbam(input_shape=(256, 256, 3))
        model.load_weights(model_path)
        model.compile(
            optimizer='adam',
            loss=combined_loss_weighted,
            metrics=['accuracy', dice_metric]
        )

        model.summary(print_fn=lambda x: None)
        return model

@st.cache_resource
def load_classification_models(inception_path, xception_path):
    """Load both classification models"""
    try:
        inception_model = load_model(inception_path, compile=False)
        xception_model = load_model(xception_path, compile=False)
        return inception_model, xception_model
    except Exception as e:
        st.error(f"Error loading classification models: {str(e)}")
        return None, None

# Preprocessing functions
def preprocess_for_segmentation(image, target_size=(256, 256)):
    """Preprocess image for segmentation - matching training preprocessing exactly"""
    # Resize image
    image = image.resize(target_size)
    image_array = np.array(image)
    
    # Ensure 3 channels
    if len(image_array.shape) == 2:
        image_array = np.stack([image_array] * 3, axis=-1)
    elif image_array.shape[-1] == 4:
        image_array = image_array[:, :, :3]
    
    # Convert to float32 (keep in 0-255 range)
    image_array = image_array.astype(np.float32)
    
    # Apply EfficientNet preprocessing (this normalizes to model's expected range)
    image_array = preprocess_input(image_array)
    
    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)
    
    return image_array

def preprocess_for_classification(original_img, segmented_mask, target_size=(256, 256)):
    """Preprocess original and segmented images for classification - matching training preprocessing exactly"""
    # Resize original fundus image to target size
    fundus_resized = original_img.resize(target_size)
    fundus_array = np.array(fundus_resized).astype(np.float32)
    # Keep in 0-255 range, no normalization (model does it internally)
    
    # Process segmented mask
    if isinstance(segmented_mask, np.ndarray):
        if segmented_mask.ndim == 3:
            segmented_mask = segmented_mask[:, :, 0]
    
    # Convert binary mask (0-1) to 0-255 range to match training
    mask_uint8 = (segmented_mask * 255).astype(np.uint8)
    mask_pil = Image.fromarray(mask_uint8)
    mask_resized = mask_pil.resize(target_size, resample=Image.NEAREST)
    
    # Convert to array, keep as 0-255 range
    seg_array = np.array(mask_resized).astype(np.float32)
    
    # Expand to 3 channels (same as training: channels=3)
    if seg_array.ndim == 2:
        seg_array = np.stack([seg_array] * 3, axis=-1)
    
    # Clip to ensure valid range (matching training preprocessing)
    fundus_array = np.clip(fundus_array, 0, 255)
    seg_array = np.clip(seg_array, 0, 255)
    
    # Add batch dimension
    fundus_batch = np.expand_dims(fundus_array, axis=0)
    seg_batch = np.expand_dims(seg_array, axis=0)
    
    return fundus_batch, seg_batch

def ensemble_predict(inception_model, xception_model, original_input, segmented_input):
    """Make ensemble prediction from both models"""
    # Get predictions from both models
    inception_pred = inception_model.predict([original_input, segmented_input], verbose=0)
    xception_pred = xception_model.predict([original_input, segmented_input], verbose=0)
    
    # Average predictions
    ensemble_pred = (inception_pred + xception_pred) / 2.0
    
    return ensemble_pred, inception_pred, xception_pred

def adaptive_threshold(mask, mean_confidence):
    """
    Automatically determine optimal threshold based on mask confidence.
    High confidence -> use standard threshold (0.5)
    Low confidence -> use lower threshold to capture vessels
    """
    if mean_confidence >= 0.6:
        # High confidence - use standard threshold
        threshold = 0.5
        confidence_level = "High"
    elif mean_confidence >= 0.4:
        # Medium-high confidence
        threshold = 0.4
        confidence_level = "Medium-High"
    elif mean_confidence >= 0.25:
        # Medium confidence - lower threshold
        threshold = 0.3
        confidence_level = "Medium"
    elif mean_confidence >= 0.15:
        # Low confidence - much lower threshold
        threshold = 0.2
        confidence_level = "Low"
    else:
        # Very low confidence - minimal threshold
        threshold = 0.15
        confidence_level = "Very Low"
    
    return threshold, confidence_level

def create_overlay(original_image, mask, alpha=0.5):
    """Create overlay of mask on original image"""
    original_resized = original_image.resize((256, 256))
    original_array = np.array(original_resized).astype(np.float32)
    
    mask_colored = np.zeros((256, 256, 3), dtype=np.float32)
    mask_colored[:, :, 1] = mask * 255
    
    overlay = (original_array * (1 - alpha) + mask_colored * alpha).astype(np.uint8)
    
    return overlay

# Streamlit UI
st.title("👁️ Fundus Image Analysis System")
st.markdown("### Automated Segmentation and Disease Classification")

# Sidebar
with st.sidebar:
    st.header("⚙️ Model Configuration")
    
    st.subheader("Segmentation Model")

    seg_model_path = hf_hub_download(
    repo_id="Hridyanshh/retinal-models",
    filename="segmentation.keras"
)

    
    seg_load_method = st.radio(
        "Load Method",
        options=['direct', 'weights'],
        index=0
    )
    
    st.subheader("Classification Models")
    inception_path = hf_hub_download(
        repo_id="Hridyanshh/retinal-models",
        filename="best_inceptionv3_fusion.keras"
    )

    xception_path = hf_hub_download(
        repo_id="Hridyanshh/retinal-models",
        filename="best_xception_fusion.keras"
    )

    
    st.divider()
    
    st.subheader("Processing Settings")
    use_adaptive = st.checkbox("Use Adaptive Thresholding", value=True, 
                                help="Automatically adjust threshold based on segmentation confidence")
    
    if not use_adaptive:
        threshold = st.slider(
            "Manual Segmentation Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05
        )
    else:
        st.info("Threshold will be automatically determined based on image quality")
    
    show_overlay = st.checkbox("Show Overlay", value=True)
    overlay_alpha = st.slider("Overlay Transparency", 0.0, 1.0, 0.5, 0.05)

# Load all models
st.info("Loading models...")
try:
    with st.spinner("Loading segmentation model..."):
        seg_model = load_segmentation_model(seg_model_path, seg_load_method)

    st.success("✅ Segmentation model loaded!")
except Exception as e:
    st.error(f"❌ Error loading segmentation model: {str(e)}")
    st.stop()

try:
    inception_model, xception_model = load_classification_models(inception_path, xception_path)
    if inception_model is not None and xception_model is not None:
        st.success("✅ Classification models loaded!")
    else:
        st.error("❌ Failed to load classification models")
        st.stop()
except Exception as e:
    st.error(f"❌ Error loading classification models: {str(e)}")
    st.stop()

# File uploader
uploaded_file = st.file_uploader(
    "Choose a fundus image...",
    type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
    help="Upload a fundus retinal image"
)

if uploaded_file is not None:
    # Load original image
    original_image = Image.open(uploaded_file).convert('RGB')
    
    # Display original image
    st.subheader("📸 Original Image")
    col_orig = st.columns([1, 2, 1])
    with col_orig[1]:
        st.image(original_image, use_container_width=True)
    
    # Process button
    if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
        
        # STEP 1: SEGMENTATION
        with st.spinner("Step 1/2: Segmenting blood vessels..."):
            preprocessed_seg = preprocess_for_segmentation(original_image)
            prediction = seg_model.predict(preprocessed_seg, verbose=0)
            mask = prediction[0, :, :, 0]
            
            # Calculate mean confidence
            mean_confidence = np.mean(mask)
            
            # Determine threshold (adaptive or manual)
            if use_adaptive:
                optimal_threshold, confidence_level = adaptive_threshold(mask, mean_confidence)
                st.info(f"🎯 Adaptive Threshold: {optimal_threshold:.2f} (Confidence Level: {confidence_level})")
            else:
                optimal_threshold = threshold
                confidence_level = "Manual"
            
            # Apply threshold
            binary_mask = (mask > optimal_threshold).astype(np.float32)
        
        st.success("✅ Segmentation complete!")
        
        # Display segmentation results
        st.subheader("🎯 Segmentation Results")
        seg_col1, seg_col2, seg_col3 = st.columns(3)
        
        with seg_col1:
            st.markdown("**Original**")
            st.image(original_image, use_container_width=True)
        
        with seg_col2:
            st.markdown("**Segmentation Mask**")
            st.image(binary_mask, use_container_width=True, clamp=True)
        
        with seg_col3:
            st.markdown("**Overlay**")
            if show_overlay:
                overlay = create_overlay(original_image, binary_mask, overlay_alpha)
                st.image(overlay, use_container_width=True)
            else:
                st.info("Enable overlay in settings")
        
        # Segmentation metrics
        seg_metric_col1, seg_metric_col2, seg_metric_col3, seg_metric_col4 = st.columns(4)
        with seg_metric_col1:
            vessel_percentage = np.sum(binary_mask) / (256 * 256) * 100
            st.metric("Vessel Coverage", f"{vessel_percentage:.2f}%")
        with seg_metric_col2:
            st.metric("Mean Confidence", f"{mean_confidence:.3f}")
        with seg_metric_col3:
            max_confidence = np.max(mask)
            st.metric("Max Confidence", f"{max_confidence:.3f}")
        with seg_metric_col4:
            st.metric("Threshold Used", f"{optimal_threshold:.2f}")
        
        # Confidence warning
        if mean_confidence < 0.3:
            st.warning(f"⚠️ Low confidence detected ({mean_confidence:.3f}). The image may be of poor quality or significantly different from training data. Results may be less reliable.")
        
        st.divider()
        
        # STEP 2: CLASSIFICATION
        with st.spinner("Step 2/2: Classifying disease..."):
            original_input, segmented_input = preprocess_for_classification(
                original_image, binary_mask
            )
            
            # Debug information
            with st.expander("🔍 Debug Information"):
                st.write(f"**Segmentation Debug:**")
                st.write(f"- Raw mask range: [{mask.min():.3f}, {mask.max():.3f}]")
                st.write(f"- Mean confidence: {mean_confidence:.3f}")
                st.write(f"- Threshold used: {optimal_threshold:.2f} ({confidence_level})")
                st.write(f"- Binary mask unique values: {np.unique(binary_mask)}")
                st.write(f"- Vessel pixel count: {np.sum(binary_mask):.0f} / {256*256}")
                
                st.write(f"\n**Classification Debug:**")
                st.write(f"- Original input shape: {original_input.shape}")
                st.write(f"- Original input range: [{original_input.min():.1f}, {original_input.max():.1f}] (expected: 0-255)")
                st.write(f"- Segmented input shape: {segmented_input.shape}")
                st.write(f"- Segmented input range: [{segmented_input.min():.1f}, {segmented_input.max():.1f}] (expected: 0-255)")
                st.write(f"- Note: Models apply their own preprocessing (InceptionV3/Xception preprocess_input)")
            
            ensemble_pred, inception_pred, xception_pred = ensemble_predict(
                inception_model, xception_model, original_input, segmented_input
            )
        
        st.success("✅ Classification complete!")
        
        # Display classification results
        st.subheader("🏥 Disease Classification Results")
        
        # Get predictions
        ensemble_class_idx = np.argmax(ensemble_pred[0])
        ensemble_class_name = DISEASE_CLASSES[ensemble_class_idx]
        ensemble_confidence = ensemble_pred[0][ensemble_class_idx] * 100
        
        # Display main prediction
        result_col1, result_col2 = st.columns([1, 2])
        
        with result_col1:
            st.markdown("### Diagnosis")
            
            # Color-code based on condition
            if ensemble_class_name == 'normal':
                st.success(f"### ✅ {ensemble_class_name.upper().replace('_', ' ')}")
            elif ensemble_class_name == 'amd':
                st.warning(f"### ⚠️ {ensemble_class_name.upper()}")
            else:
                st.error(f"### ⚠️ {ensemble_class_name.upper().replace('_', ' ')}")
            
            st.metric("Confidence", f"{ensemble_confidence:.2f}%")
        
        with result_col2:
            st.markdown("### Probability Distribution")
            
            # Create a DataFrame for visualization
            import pandas as pd
            prob_data = {
                'Disease': DISEASE_CLASSES,
                'Ensemble': ensemble_pred[0] * 100,
                'InceptionV3': inception_pred[0] * 100,
                'Xception': xception_pred[0] * 100
            }
            
            # Display as bar chart
            df = pd.DataFrame(prob_data)
            df_display = df.set_index('Disease')
            st.bar_chart(df_display)
        
        # Detailed probabilities
        st.subheader("📊 Detailed Predictions")
        
        prob_col1, prob_col2, prob_col3 = st.columns(3)
        
        with prob_col1:
            st.markdown("**Ensemble Model**")
            for i, disease in enumerate(DISEASE_CLASSES):
                st.write(f"{disease.replace('_', ' ').title()}: {ensemble_pred[0][i]*100:.2f}%")
        
        with prob_col2:
            st.markdown("**InceptionV3 Model**")
            for i, disease in enumerate(DISEASE_CLASSES):
                st.write(f"{disease.replace('_', ' ').title()}: {inception_pred[0][i]*100:.2f}%")
        
        with prob_col3:
            st.markdown("**Xception Model**")
            for i, disease in enumerate(DISEASE_CLASSES):
                st.write(f"{disease.replace('_', ' ').title()}: {xception_pred[0][i]*100:.2f}%")
        
        # Clinical interpretation
        st.divider()
        st.subheader("🩺 Clinical Interpretation")
        
        if ensemble_class_name == 'amd':
            st.warning("""
            **Age-related Macular Degeneration (AMD) Detected**
            - Progressive deterioration of the macula, central part of the retina
            - Can lead to loss of central vision
            - Vessel coverage: {:.2f}%
            - Recommendation: Immediate retinal specialist consultation required
            - Management: Anti-VEGF injections, vitamin supplements, lifestyle modifications
            """.format(vessel_percentage))
        elif ensemble_class_name == 'diabetic_retinopathy':
            st.warning("""
            **Diabetic Retinopathy Detected**
            - This condition is caused by damage to blood vessels in the retina due to diabetes
            - Vessel coverage: {:.2f}% indicates vascular changes
            - Recommendation: Immediate ophthalmologist consultation required
            - Management: Blood sugar control, regular monitoring, possible laser treatment
            """.format(vessel_percentage))
        elif ensemble_class_name == 'glaucoma':
            st.warning("""
            **Glaucoma Suspected**
            - Progressive optic nerve damage condition
            - May lead to vision loss if untreated
            - Recommendation: Comprehensive eye examination including IOP measurement
            - Management: Medications, laser treatment, or surgery may be needed
            """)
        elif ensemble_class_name == 'normal':
            st.success("""
            **Normal Fundus**
            - No significant pathological findings detected
            - Vessel coverage: {:.2f}% is within normal range
            - Recommendation: Continue routine annual eye examinations
            - Maintain good eye health practices
            """.format(vessel_percentage))
        
        # Download section
        st.divider()
        st.subheader("💾 Download Results")
        
        download_col1, download_col2, download_col3 = st.columns(3)
        
        with download_col1:
            mask_image = Image.fromarray((binary_mask * 255).astype(np.uint8))
            mask_buffer = io.BytesIO()
            mask_image.save(mask_buffer, format='PNG')
            st.download_button(
                label="⬇️ Download Mask",
                data=mask_buffer.getvalue(),
                file_name="segmentation_mask.png",
                mime="image/png"
            )
        
        with download_col2:
            if show_overlay:
                overlay_image = Image.fromarray(overlay.astype(np.uint8))
                overlay_buffer = io.BytesIO()
                overlay_image.save(overlay_buffer, format='PNG')
                st.download_button(
                    label="⬇️ Download Overlay",
                    data=overlay_buffer.getvalue(),
                    file_name="overlay.png",
                    mime="image/png"
                )
        
        with download_col3:
            # Create report
            report = f"""FUNDUS IMAGE ANALYSIS REPORT
================================

DIAGNOSIS: {ensemble_class_name.upper().replace('_', ' ')}
Confidence: {ensemble_confidence:.2f}%

SEGMENTATION METRICS:
- Vessel Coverage: {vessel_percentage:.2f}%
- Mean Confidence: {mean_confidence:.3f}
- Max Confidence: {max_confidence:.3f}
- Threshold Used: {optimal_threshold:.2f} ({confidence_level})

CLASSIFICATION PROBABILITIES:
Ensemble Model:
{chr(10).join([f"  - {d.replace('_', ' ').title()}: {ensemble_pred[0][i]*100:.2f}%" for i, d in enumerate(DISEASE_CLASSES)])}

InceptionV3 Model:
{chr(10).join([f"  - {d.replace('_', ' ').title()}: {inception_pred[0][i]*100:.2f}%" for i, d in enumerate(DISEASE_CLASSES)])}

Xception Model:
{chr(10).join([f"  - {d.replace('_', ' ').title()}: {xception_pred[0][i]*100:.2f}%" for i, d in enumerate(DISEASE_CLASSES)])}

NOTE: This is an automated analysis and should not replace professional medical diagnosis.
"""
            st.download_button(
                label="⬇️ Download Report",
                data=report,
                file_name="analysis_report.txt",
                mime="text/plain"
            )

else:
    st.info("👆 Please upload a fundus image to begin analysis")
    
    # Information section
    with st.expander("ℹ️ System Information"):
        st.markdown("""
        ### 🔬 Analysis Pipeline
        
        **Step 1: Segmentation**
        - Model: U-Net with EfficientNetB0 backbone + CBAM
        - Purpose: Extract blood vessel structures
        - Input: 256×256 RGB fundus image
        - Output: Binary segmentation mask
        
        **Step 2: Classification**
        - Models: InceptionV3 + Xception (Ensemble)
        - Input: Original image + Segmented mask
        - Output: Disease classification
        - Classes: AMD, Diabetic Retinopathy, Glaucoma, Normal
        
        ### 📋 How to Use
        1. Upload a fundus retinal image
        2. Adjust segmentation threshold if needed
        3. Click "Analyze Image" to process
        4. Review segmentation and classification results
        5. Download masks, overlays, and analysis report
        
        ### ⚠️ Disclaimer
        This system is designed for research and educational purposes. 
        Results should not be used as the sole basis for clinical decisions.
        Always consult with qualified healthcare professionals for medical diagnosis.
        """)

# Footer
st.divider()
st.caption("Fundus Image Analysis System | Segmentation: U-Net + EfficientNetB0 + CBAM | Classification: InceptionV3 + Xception Fusion")