import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import requests
from PIL import Image
import os
import time
import threading
import glob
from kaggle.api.kaggle_api_extended import KaggleApi
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode

# --- Custom CSS for UI Enhancement ---
custom_css = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="st-emotion-"] {
        font-family: 'Inter', sans-serif;
    }

    body {
        background: linear-gradient(to bottom right, #F9FAFB, #E5E7EB);
        min-height: 100vh;
    }

    .stApp {
        background-color: transparent;
    }

    .detection-card {
        background-color: rgba(249, 250, 251, 0.8);
        backdrop-filter: blur(10px);
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1), 0 8px 10px -6px rgba(0, 0, 0, 0.1);
        border-radius: 0.75rem;
        padding: 1.5rem;
        margin-bottom: 2rem;
    }

    .stButton > button {
        border-radius: 0.5rem;
        font-weight: 500;
        padding: 0.625rem 1.25rem;
        transition: all 0.2s ease-in-out;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .stButton > button:hover {
        opacity: 0.9;
    }

    .st-emotion-cache-1cypcdb {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    .st-emotion-cache-1r6dm1x {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    .stRadio > label {
        display: flex;
        align-items: center;
        padding: 0.75rem;
        border: 1px solid #E5E7EB;
        border-radius: 0.5rem;
        transition: all 0.2s ease-in-out;
        cursor: pointer;
    }
    .stRadio > label:hover {
        border-color: #10B981;
        background-color: rgba(16, 185, 129, 0.05);
    }
    .stRadio > label > div > p {
        margin-left: 0.75rem;
        color: #374151;
        font-weight: 500;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .stRadio [data-testid="stRadioInline"] > label > div:first-child {
        margin-top: 0 !important;
        margin-bottom: 0 !important;
    }

    .stAlert {
        border-radius: 0.5rem;
    }

    h1.st-emotion-cache-10q700h {
        font-size: 3rem;
        font-weight: 700;
        color: #1F2937;
        text-align: center;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 1rem;
        margin-bottom: 0.5rem;
    }
    h1.st-emotion-cache-10q700h .fas {
        color: #10B981;
        font-size: 3.5rem;
    }
    .st-emotion-cache-10q700h + div > p {
        color: #4B5563;
        max-width: 56rem;
        margin-left: auto;
        margin-right: auto;
        text-align: center;
        font-size: 1.125rem;
    }

    .confidence-value-display {
        background-color: #10B981;
        color: white;
        font-weight: 500;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.875rem;
    }

    .legend-item {
        display: flex;
        align-items: center;
        margin-bottom: 0.5rem;
    }
    .legend-color-box {
        width: 0.75rem;
        height: 0.75rem;
        border-radius: 9999px;
        margin-right: 0.5rem;
    }
    .legend-text {
        font-size: 0.875rem;
    }

    .status-item {
        display: flex;
        align-items: center;
        margin-bottom: 0.75rem;
    }
    .status-indicator {
        width: 0.5rem;
        height: 0.5rem;
        border-radius: 9999px;
        margin-right: 0.75rem;
    }
    .status-text {
        color: #374151;
    }

    .placeholder-box {
        border: 2px dashed #D1D5DB;
        border-radius: 0.75rem;
        display: flex;
        align-items: center;
        justify-content: center;
        height: 18rem;
        text-align: center;
        flex-direction: column;
        color: #9CA3AF;
    }
    .placeholder-box i {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    .placeholder-box p {
        color: #9CA3AF;
    }

    .stats-card {
        border-radius: 0.5rem;
        padding: 1rem;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    .stats-icon-box {
        width: 2.5rem;
        height: 2.5rem;
        border-radius: 9999px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .stats-icon-box i {
        color: white;
    }
    .stats-label {
        color: #4B5563;
        font-size: 0.875rem;
    }
    .stats-value {
        font-weight: 700;
        color: #1F2937;
        font-size: 1.25rem;
    }

    .glow-border {
        box-shadow: 0 0 15px rgba(16, 185, 129, 0.3);
    }
    .video-placeholder {
        background-color: black;
        aspect-ratio: 16 / 9;
        display: flex;
        align-items: center;
        justify-content: center;
        position: relative;
        border-radius: 1rem;
        overflow: hidden;
    }
    .video-placeholder .overlay {
        position: absolute;
        width: 100%;
        height: 100%;
        opacity: 0.2;
        background: linear-gradient(to right, #4CAF50, #2196F3);
    }
    .video-placeholder .content {
        z-index: 10;
        text-align: center;
        color: #D1D5DB;
    }
    .video-placeholder .content i {
        font-size: 4rem;
    }

    .footer-badge {
        background-color: #F3F4F6;
        border: 1px solid #E5E7EB;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    .footer-text {
        color: #4B5563;
    }
    .footer-highlight {
        color: #10B981;
        font-weight: 500;
    }

    .pulse-animation {
        animation: pulse-stream 2s infinite;
    }
    @keyframes pulse-stream {
        0% { box-shadow: 0 0 0 0 rgba(59, 130, 246, 0.4); }
        70% { box-shadow: 0 0 0 10px rgba(59, 130, 246, 0); }
        100% { box-shadow: 0 0 0 0 rgba(59, 130, 246, 0); }
    }

    [data-testid="stButton-detect_image_button"] > button {
        background-color: #3B82F6;
        color: white;
        animation: pulse 2s infinite;
    }
    [data-testid="stButton-detect_image_button"] > button:hover {
        background-color: #2563EB;
    }

    [data-testid="stButton-process_video_button"] > button {
        background-color: #3B82F6;
        color: white;
    }
    [data-testid="stButton-process_video_button"] > button:hover {
        background-color: #2563EB;
    }

    [data-testid="stButton-start_camera_button"] > button {
        background-color: #16A34A;
        color: white;
    }
    [data-testid="stButton-start_camera_button"] > button:hover {
        background-color: #15803D;
    }

    [data-testid="stButton-stop_detection_button"] > button {
        background-color: #EF4444;
        color: white;
    }
    [data-testid="stButton-stop_detection_button"] > button:hover {
        background-color: #DC2626;
    }

    [data-testid="stButton-capture_image_button"] > button {
        background-color: #F3F4F6;
        border: 1px solid #D1D5DB;
        color: #374151;
    }
    [data-testid="stButton-capture_image_button"] > button:hover {
        background-color: #E5E7EB;
    }

    [data-testid="stButton-load_kaggle_dataset_button"] > button {
        background-color: #8B5CF6;
        color: white;
    }
    [data-testid="stButton-load_kaggle_dataset_button"] > button:hover {
        background-color: #7C3AED;
    }
</style>
"""

# Inject custom CSS
st.markdown(custom_css, unsafe_allow_html=True)

# --- Kaggle Dataset Functionality ---
@st.cache_resource
def download_kaggle_dataset(dataset_slug, download_path):
    try:
        os.environ['KAGGLE_USERNAME'] = st.secrets['kaggle']['username']
        os.environ['KAGGLE_KEY'] = st.secrets['kaggle']['key']
        
        api = KaggleApi()
        api.authenticate()
        
        api.dataset_download_files(dataset_slug, path=download_path, unzip=True)
        
        valid_images_path = os.path.join(download_path, 'valid', 'images')
        image_extensions = ['*.jpg', '*.jpeg', '*.png']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(valid_images_path, ext)))
        
        image_files = image_files[:10]
        
        return image_files
    except Exception as e:
        st.error(f"Error downloading Kaggle dataset: {e}")
        return []

def load_kaggle_image(image_path):
    try:
        image = Image.open(image_path)
        return image, np.array(image)
    except Exception as e:
        st.error(f"Error loading image {image_path}: {e}")
        return None, None

# --- Model Loading ---
@st.cache_resource
def load_yolo_model(url):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as temp_model_file:
            for chunk in response.iter_content(chunk_size=8192):
                temp_model_file.write(chunk)
            model_path = temp_model_file.name

        model = YOLO(model_path)
        return model
    except requests.exceptions.RequestException as e:
        st.error(f"Error downloading the model: {e}")
        return None
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

# --- Video Transformer Class for Live Camera ---
class WeedCropDetector(VideoTransformerBase):
    def __init__(self, model, class_names, colors, confidence_threshold):
        self.model = model
        self.class_names = class_names
        self.colors = colors
        self.confidence_threshold = confidence_threshold
        if 'live_counts' not in st.session_state:
            st.session_state.live_counts = {'crop': 0, 'weed': 0}
        if 'live_counts_lock' not in st.session_state:
            st.session_state.live_counts_lock = threading.Lock()

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        results = self.model.predict(img, conf=self.confidence_threshold, verbose=False)

        annotated_img = img.copy()
        current_crop_count = 0
        current_weed_count = 0

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0]
                cls_id = int(box.cls[0])

                if cls_id < len(self.class_names):
                    class_name = self.class_names[cls_id]
                else:
                    class_name = f"Unknown {cls_id}"

                if class_name == 'crop':
                    current_crop_count += 1
                elif class_name == 'weed':
                    current_weed_count += 1

                label = f"{class_name} {conf:.2f}"
                color = self.colors.get(class_name, (255, 255, 255))

                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
                (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(annotated_img, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
                cv2.putText(annotated_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        with st.session_state.live_counts_lock:
            st.session_state.live_counts['crop'] = current_crop_count
            st.session_state.live_counts['weed'] = current_weed_count
            
        return annotated_img

# --- UI and Main Logic ---
MODEL_URL = "https://raw.githubusercontent.com/blurerjr/multidataset-weed-detection/master/best.pt"

model = None
try:
    with st.spinner("Downloading and loading the model (this may take a moment)..."):
        model = load_yolo_model(MODEL_URL)
except Exception as e:
    st.error(f"Failed to load model at startup: {e}")

if model is None:
    st.warning("Model could not be loaded. Please check the URL and your internet connection.")
    st.stop()

CLASS_NAMES = ['crop', 'weed']
COLORS = {
    'crop': (0, 255, 0),
    'weed': (0, 0, 255)
}

# --- Header Section ---
st.markdown(
    f"""
    <header class="text-center mb-12">
        <div class="flex items-center justify-center gap-4 mb-2">
            <i class="fas fa-leaf text-5xl" style="color:#10B981;"></i>
            <h1 style="font-size: 3rem; font-weight: bold; color: #1F2937; margin:0;">
                Weed <span style="color:#10B981;">&</span> Crop Detection
            </h1>
        </div>
        <p style="color:#4B5563; max-width: 56rem; margin-left: auto; margin-right: auto; font-size: 1.125rem;">
            Use computer vision to identify weeds and crops in images, videos, or live camera feed.
            Powered by YOLOv8 and Ultralytics.
        </p>
    </header>
    """,
    unsafe_allow_html=True
)

# --- Main Layout (Sidebar + Content) ---
col1, col2 = st.columns([1, 3])

with col1:
    st.markdown('<div class="detection-card">', unsafe_allow_html=True)
    st.markdown(
        """
        <h2 style="font-size: 1.5rem; font-weight: bold; color: #1F2937; margin-bottom: 1.5rem; display: flex; align-items: center; gap: 0.5rem;">
            <i class="fas fa-sliders-h" style="color:#3B82F6;"></i>
            Detection Options
        </h2>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<h3 style="font-weight: 500; color: #374151; margin-bottom: 0.5rem;">Detection Mode</h3>', unsafe_allow_html=True)
    detection_mode = st.radio(
        "Select Detection Mode",
        ["Image", "Video", "Live Camera"],
        key="detection_mode_radio",
        label_visibility="collapsed"
    )

    st.markdown('<div style="margin-bottom: 1.5rem;"></div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
            <h3 style="font-weight: 500; color: #374151; margin:0;">Confidence Threshold</h3>
            <span id="confidenceValue" class="confidence-value-display"></span>
        </div>
        """,
        unsafe_allow_html=True
    )
    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.0, max_value=1.0, value=0.05, step=0.05,
        key="confidence_slider",
        label_visibility="collapsed"
    )
    st.markdown(f"<script>document.getElementById('confidenceValue').textContent = '{confidence_threshold:.2f}';</script>", unsafe_allow_html=True)

    st.markdown('<div style="margin-bottom: 1.5rem;"></div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div style="padding-top: 1rem; border-top: 1px solid #E5E7EB;">
            <h3 style="font-weight: 500; color: #374151; margin-bottom: 0.5rem;">Detection Legend</h3>
            <div class="legend-item">
                <div class="legend-color-box" style="background-color: #10B981;"></div>
                <span class="legend-text">Crop</span>
            </div>
            <div class="legend-item">
                <div class="legend-color-box" style="background-color: #EF4444;"></div>
                <span class="legend-text">Weed</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="detection-card" style="margin-top: 1.5rem;">', unsafe_allow_html=True)
    st.markdown(
        """
        <h3 style="font-size: 1.25rem; font-weight: bold; color: #1F2937; margin-bottom: 1rem; display: flex; align-items: center; gap: 0.5rem;">
            <i class="fas fa-info-circle" style="color:#3B82F6;"></i> System Status
        </h3>
        """,
        unsafe_allow_html=True
    )
    st.markdown(
        f"""
        <div class="status-item">
            <div class="status-indicator {'bg-green-500 animate-pulse' if model else 'bg-red-500'}"></div>
            <span class="status-text">Model: {'Loaded' if model else 'Failed to Load'}</span>
        </div>
        <div class="status-item">
            <div class="status-indicator {'bg-green-500 animate-pulse' if model else 'bg-red-500'}"></div>
            <span class="status-text">Detection: Ready</span>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    if detection_mode == "Image":
        st.markdown('<div class="detection-card">', unsafe_allow_html=True)
        st.markdown(
            """
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1.5rem;">
                <h2 style="font-size: 1.5rem; font-weight: bold; color: #1F2937; display: flex; align-items: center; gap: 0.5rem;">
                    <i class="fas fa-image" style="color:#3B82F6;"></i> Image Detection
                </h2>
            </div>
            """,
            unsafe_allow_html=True
        )

        col_upload, col_kaggle = st.columns([1, 1])
        
        with col_upload:
            uploaded_file = st.file_uploader(
                "Upload Image",
                type=["jpg", "jpeg", "png"],
                key="image_uploader",
                label_visibility="collapsed"
            )
        
        with col_kaggle:
            load_kaggle_dataset = st.button("Load Kaggle Dataset", key="load_kaggle_dataset_button")
        
        if 'kaggle_images' not in st.session_state:
            st.session_state.kaggle_images = []
        if 'selected_kaggle_image' not in st.session_state:
            st.session_state.selected_kaggle_image = None
        
        if load_kaggle_dataset:
            with st.spinner("Downloading Kaggle dataset..."):
                dataset_slug = "cubeai/crop-weed-detection-for-yolov8"
                download_path = tempfile.mkdtemp()
                st.session_state.kaggle_images = download_kaggle_dataset(dataset_slug, download_path)
                if st.session_state.kaggle_images:
                    st.success(f"Loaded {len(st.session_state.kaggle_images)} images from Kaggle dataset (valid/images).")
                else:
                    st.warning("No images found in the valid/images folder or an error occurred.")
        
        if st.session_state.kaggle_images:
            image_names = [os.path.basename(path) for path in st.session_state.kaggle_images]
            selected_image_name = st.selectbox(
                "Select an image from Kaggle dataset",
                options=["Select an image"] + image_names,
                index=0,
                key="kaggle_image_selectbox"
            )
            if selected_image_name != "Select an image":
                selected_image_path = st.session_state.kaggle_images[image_names.index(selected_image_name)]
                st.session_state.selected_kaggle_image = selected_image_path
        
        image_col, processed_image_col = st.columns(2)
        
        original_image_placeholder = image_col.empty()
        processed_image_placeholder = processed_image_col.empty()
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            img_array = np.array(image)
            original_image_placeholder.image(image, caption="Uploaded Image", use_container_width=True)
        elif st.session_state.selected_kaggle_image is not None:
            image, img_array = load_kaggle_image(st.session_state.selected_kaggle_image)
            if image is not None:
                original_image_placeholder.image(image, caption="Selected Kaggle Image", use_container_width=True)
        else:
            original_image_placeholder.markdown(
                """
                <h3 style="font-weight: 500; color: #374151; margin-bottom: 0.5rem; display: flex; align-items: center; gap: 0.5rem;">
                    <i class="fas fa-file-image" style="color:#6B7280;"></i> Original Image
                </h3>
                <div class="placeholder-box">
                    <i class="fas fa-image"></i>
                    <p>Upload an image or select from Kaggle dataset to get started</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            processed_image_placeholder.markdown(
                """
                <h3 style="font-weight: 500; color: #374151; margin-bottom: 0.5rem; display: flex; align-items: center; gap: 0.5rem;">
                    <i class="fas fa-tags" style="color:#6B7280;"></i> Processed Image
                </h3>
                <div class="placeholder-box">
                    <i class="fas fa-project-diagram"></i>
                    <p>Detection results will appear here</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        detect_button_clicked = st.button("Detect Weeds & Crops", key="detect_image_button")
        
        if detect_button_clicked and (uploaded_file is not None or st.session_state.selected_kaggle_image is not None):
            with st.spinner("Processing image..."):
                if uploaded_file is not None:
                    image = Image.open(uploaded_file)
                    img_array = np.array(image)
                elif st.session_state.selected_kaggle_image is not None:
                    image, img_array = load_kaggle_image(st.session_state.selected_kaggle_image)
                
                if img_array is not None:
                    results = model.predict(img_array, conf=confidence_threshold)
                    annotated_img_array = img_array.copy()
                    detection_count = {'crop': 0, 'weed': 0}
                    
                    for result in results:
                        for box in result.boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            conf = box.conf[0]
                            cls_id = int(box.cls[0])
                            
                            if cls_id < len(CLASS_NAMES):
                                class_name = CLASS_NAMES[cls_id]
                            else:
                                class_name = f"Unknown {cls_id}"
                            
                            detection_count[class_name] += 1
                            
                            label = f"{class_name} {conf:.2f}"
                            color = COLORS.get(class_name, (255, 255, 255))
                            
                            cv2.rectangle(annotated_img_array, (x1, y1), (x2, y2), color, 2)
                            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                            cv2.rectangle(annotated_img_array, (x1, y1 - h - 5), (x1 + w, y1), color, -1)
                            cv2.putText(annotated_img_array, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                    
                    processed_image_placeholder.image(annotated_img_array, caption="Processed Image", use_container_width=True)
                    st.success(f"Detection complete! Found {detection_count['crop']} crops and {detection_count['weed']} weeds.")
                    
                    col_crop_stats, col_weed_stats = st.columns(2)
                    with col_crop_stats:
                        st.markdown(
                            f"""
                            <div class="stats-card" style="background-color:#ECFDF5; border:1px solid #D1FAE5;">
                                <div class="stats-icon-box" style="background-color:#10B981;">
                                    <i class="fas fa-leaf"></i>
                                </div>
                                <div>
                                    <p class="stats-label">Crop Detection</p>
                                    <p class="stats-value">{detection_count['crop']}</p>
                                </div>
                            </div>
                            """, unsafe_allow_html=True
                        )
                    with col_weed_stats:
                        st.markdown(
                            f"""
                            <div class="stats-card" style="background-color:#FEF2F2; border:1px solid #FEE2E2;">
                                <div class="stats-icon-box" style="background-color:#EF4444;">
                                    <i class="fas fa-tree"></i>
                                </div>
                                <div>
                                    <p class="stats-label">Weed Detection</p>
                                    <p class="stats-value">{detection_count['weed']}</p>
                                </div>
                            </div>
                            """, unsafe_allow_html=True
                        )
        elif detect_button_clicked:
            st.warning("Please upload an image or select an image from the Kaggle dataset before clicking 'Detect Weeds & Crops'.")
        
        st.markdown(
            """
            <div style="margin-top: 1.5rem; display: flex; align-items: center; gap: 1rem;">
                <div style="background-color:#FFFBEB; border:1px solid #FDE68A; border-radius:0.5rem; padding:0.5rem 1rem; color:#92400E; font-size:0.875rem; flex:1; display:flex; align-items:center; gap:0.25rem;">
                    <i class="fas fa-info-circle"></i> Upload an image or select from Kaggle dataset to process.
                </div>
            </div>
            """, unsafe_allow_html=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

    elif detection_mode == "Video":
        st.markdown('<div class="detection-card">', unsafe_allow_html=True)
        st.markdown(
            """
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1.5rem;">
                <h2 style="font-size: 1.5rem; font-weight: bold; color: #1F2937; display: flex; align-items: center; gap: 0.5rem;">
                    <i class="fas fa-video" style="color:#3B82F6;"></i> Video Detection
                </h2>
            </div>
            """,
            unsafe_allow_html=True
        )

        uploaded_file = st.file_uploader(
            "Upload Video",
            type=["mp4", "mov", "avi"],
            key="video_uploader",
            label_visibility="collapsed"
        )

        video_placeholder = st.empty()
        if uploaded_file is None:
            video_placeholder.markdown(
                """
                <div class="placeholder-box" style="height: 24rem;">
                    <i class="fas fa-film" style="font-size: 3.5rem; margin-bottom: 0.75rem;"></i>
                    <p style="color:#4B5563;">Upload a video to process weed and crop detection</p>
                    <p style="font-size: 0.875rem; color:#6B7280; margin-top: 0.5rem;">Supports MP4, MOV, AVI formats</p>
                </div>
                """, unsafe_allow_html=True
            )
        else:
            video_placeholder.video(uploaded_file, format="video/mp4", start_time=0)

        process_video_button = st.button("Process Video", key="process_video_button")

        if uploaded_file is not None and process_video_button:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_file.read())
            video_path = tfile.name
            tfile.close()

            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                st.error("Error: Could not open video file.")
                os.remove(video_path)
            else:
                st.info("Processing video... This may take a while depending on video length.")
                stframe = st.empty()
                progress_bar = st.progress(0)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                current_frame_idx = 0

                total_crops = 0
                total_weeds = 0

                status_col1, status_col2 = st.columns(2)
                with status_col1:
                    progress_text = st.empty()
                with status_col2:
                    video_stats_placeholder = st.empty()

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    results = model.predict(frame, conf=confidence_threshold, verbose=False)

                    annotated_frame = frame.copy()
                    
                    frame_crops = 0
                    frame_weeds = 0

                    for result in results:
                        for box in result.boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            conf = box.conf[0]
                            cls_id = int(box.cls[0])

                            if cls_id < len(CLASS_NAMES):
                                class_name = CLASS_NAMES[cls_id]
                            else:
                                class_name = f"Unknown {cls_id}"
                            
                            if class_name == 'crop':
                                frame_crops += 1
                            elif class_name == 'weed':
                                frame_weeds += 1

                            label = f"{class_name} {conf:.2f}"
                            color = COLORS.get(class_name, (255, 255, 255))

                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                            cv2.rectangle(annotated_frame, (x1, y1 - h - 5), (x1 + w, y1), color, -1)
                            cv2.putText(annotated_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

                    stframe.image(annotated_frame, channels="BGR", use_container_width=True)

                    current_frame_idx += 1
                    progress = min(current_frame_idx / frame_count, 1.0)
                    progress_bar.progress(progress)
                    progress_text.markdown(f'<p style="color:#4B5563; font-size:0.875rem; margin-top:0.5rem;">{int(progress*100)}% completed</p>', unsafe_allow_html=True)

                    total_crops += frame_crops
                    total_weeds += frame_weeds

                    video_stats_placeholder.markdown(
                        f"""
                        <h3 style="font-weight: 500; color: #374151; margin-bottom: 0.5rem;">Detection Stats (Current Frame)</h3>
                        <div style="display:flex; gap:1rem;">
                            <div style="text-align:center;">
                                <p style="color:#4B5563; font-size:0.875rem;">Crops</p>
                                <p style="font-weight:bold; color:#1F2937; font-size:1.25rem;">{frame_crops}</p>
                            </div>
                            <div style="text-align:center;">
                                <p style="color:#4B5563; font-size:0.875rem;">Weeds</p>
                                <p style="font-weight:bold; color:#1F2937; font-size:1.25rem;">{frame_weeds}</p>
                            </div>
                            <div style="text-align:center;">
                                <p style="color:#4B5563; font-size:0.875rem;">FPS</p>
                                <p style="font-weight:bold; color:#1F2937; font-size:1.25rem;">{int(fps)}</p>
                            </div>
                        </div>
                        """, unsafe_allow_html=True
                    )

                cap.release()
                os.remove(video_path)
                st.success(f"Video processing complete! Total Crops: {total_crops}, Total Weeds: {total_weeds}")
        elif process_video_button and uploaded_file is None:
            st.warning("Please upload a video before clicking 'Process Video'.")
        
        st.markdown(
            """
            <div style="margin-top: 1.5rem; display: flex; align-items: center; gap: 1rem;">
                <div style="background-color:#EFF6FF; border:1px solid #DBEAFE; border-radius:0.5rem; padding:0.5rem 1rem; color:#1E40AF; font-size:0.875rem; flex:1; display:flex; align-items:center; gap:0.25rem;">
                    <i class="fas fa-info-circle"></i> Processing may take time depending on video length.
                </div>
            </div>
            """, unsafe_allow_html=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

    elif detection_mode == "Live Camera":
        st.markdown('<div class="detection-card">', unsafe_allow_html=True)
        st.markdown(
            """
            <h2 style="font-size: 1.5rem; font-weight: bold; color: #1F2937; margin-bottom: 1.5rem; display: flex; align-items: center; gap: 0.5rem;">
                <i class="fas fa-camera" style="color:#3B82F6;"></i> Live Camera Detection
            </h2>
            """,
            unsafe_allow_html=True
        )

        live_status_col, live_detection_col = st.columns(2)
        camera_status_placeholder = live_status_col.empty()
        active_detections_placeholder = live_detection_col.empty()

        if 'live_counts' not in st.session_state:
            st.session_state.live_counts = {'crop': 0, 'weed': 0}
        if 'live_counts_lock' not in st.session_state:
            st.session_state.live_counts_lock = threading.Lock()

        webrtc_ctx = webrtc_streamer(
            key="weed_detection_live_stream",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration={
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
            },
            video_transformer_factory=lambda: WeedCropDetector(model, CLASS_NAMES, COLORS, confidence_threshold),
            media_stream_constraints={"video": True, "audio": False},
            async_transform=True,
        )

        col_start, col_stop, col_capture = st.columns([1, 1, 2])
        with col_start:
            start_button = st.button("Start Camera", key="start_camera_button")
        with col_stop:
            stop_button = st.button("Stop Detection", key="stop_detection_button")
        with col_capture:
            capture_button = st.button("Capture Image", key="capture_image_button")

        if webrtc_ctx.video_transformer:
            camera_status_placeholder.markdown(
                """
                <h3 style="font-weight: 500; color: #374151; margin-bottom: 0.5rem;">Camera Status</h3>
                <div style="display: flex; align-items: center;">
                    <div class="status-indicator bg-green-500 pulse-animation"></div>
                    <span class="status-text">Active</span>
                </div>
                """, unsafe_allow_html=True
            )
            while True:
                with st.session_state.live_counts_lock:
                    crop_count = st.session_state.live_counts['crop']
                    weed_count = st.session_state.live_counts['weed']

                active_detections_placeholder.markdown(
                    f"""
                    <h3 style="font-weight: 500; color: #374151; margin-bottom: 0.5rem;">Active Detections</h3>
                    <div style="display:flex; gap:1rem;">
                        <div style="text-align:center;">
                            <p style="color:#4B5563; font-size:0.875rem;">Crops</p>
                            <p style="font-weight:bold; color:#1F2937; font-size:1.25rem;">{crop_count}</p>
                        </div>
                        <div style="text-align:center;">
                            <p style="color:#4B5563; font-size:0.875rem;">Weeds</p>
                            <p style="font-weight:bold; color:#1F2937; font-size:1.25rem;">{weed_count}</p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True
                )
                time.sleep(0.5)
        else:
            camera_status_placeholder.markdown(
                """
                <h3 style="font-weight: 500; color: #374151; margin-bottom: 0.5rem;">Camera Status</h3>
                <div style="display: flex; align-items: center;">
                    <div class="status-indicator bg-red-500"></div>
                    <span class="status-text">Not Active</span>
                </div>
                """, unsafe_allow_html=True
            )
            active_detections_placeholder.markdown(
                """
                <h3 style="font-weight: 500; color: #374151; margin-bottom: 0.5rem;">Active Detections</h3>
                <div style="display:flex; gap:1rem;">
                    <div style="text-align:center;">
                        <p style="color:#4B5563; font-size:0.875rem;">Crops</p>
                        <p style="font-weight:bold; color:#1F2937; font-size:1.25rem;">0</p>
                    </div>
                    <div style="text-align:center;">
                        <p style="color:#4B5563; font-size:0.875rem;">Weeds</p>
                        <p style="font-weight:bold; color:#1F2937; font-size:1.25rem;">0</p>
                    </div>
                </div>
                """, unsafe_allow_html=True
            )

        st.markdown(
            """
            <div style="margin-top: 2rem; background-color:#EFF6FF; border:1px solid #DBEAFE; border-radius:0.5rem; padding:1rem;">
                <div style="display:flex; align-items:flex-start; gap:0.75rem;">
                    <i class="fas fa-info-circle" style="color:#3B82F6; font-size:1.25rem; margin-top:0.25rem;"></i>
                    <p style="color:#1E40AF;">
                        Click "Start Camera" and grant camera access to begin real-time weed and crop detection.
                        The system will process each frame and overlay detection information in real-time.
                    </p>
                </div>
            </div>
            """, unsafe_allow_html=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

# --- Footer Section ---
st.markdown(
    """
    <footer style="margin-top: 3rem; padding-top: 1.5rem; border-top: 1px solid #E5E7EB; text-align: center; color: #4B5563;">
        <div style="display: flex; items-align: center; justify-content: center; gap: 1rem; margin-bottom: 0.75rem;">
            <div class="footer-badge">
                <span style="color:#10B981; font-weight:bold;">YOLO</span>v8
            </div>
            <div class="footer-badge">
                <span style="color:#3B82F6; font-weight:bold;">Streamlit</span>
            </div>
        </div>
        <p class="footer-text">
            Developed by <span class="footer-highlight">blurerjr/mu</span> using Ultralytics YOLO and Streamlit.
        </p>
    </footer>
    """,
    unsafe_allow_html=True
)
