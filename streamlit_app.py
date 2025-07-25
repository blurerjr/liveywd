import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import requests
from PIL import Image
import os
import time # For video processing progress simulation

# Import necessary components for live camera streaming
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode

# --- Custom CSS for UI Enhancement ---
# This CSS mimics the Tailwind and custom styles from your HTML guidance.
# Streamlit's default components don't directly support Tailwind classes,
# so we inject custom CSS for a similar look and feel.
custom_css = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="st-emotion-"] {
        font-family: 'Inter', sans-serif;
    }

    body {
        background: linear-gradient(to bottom right, #F9FAFB, #E5E7EB); /* bg-gradient-to-br from-gray-50 to-gray-100 */
        min-height: 100vh;
    }

    .stApp {
        background-color: transparent; /* Ensure Streamlit app background is transparent to show body gradient */
    }

    /* Main container styling */
    .detection-card {
        background-color: rgba(249, 250, 251, 0.8); /* bg-light with transparency */
        backdrop-filter: blur(10px);
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1), 0 8px 10px -6px rgba(0, 0, 0, 0.1); /* shadow-xl */
        border-radius: 0.75rem; /* rounded-xl */
        padding: 1.5rem; /* p-6 */
        margin-bottom: 2rem; /* mb-8 */
    }

    /* Specific styling for Streamlit elements to match the HTML design */
    .stButton > button {
        border-radius: 0.5rem; /* rounded-lg */
        font-weight: 500; /* font-medium */
        padding: 0.625rem 1.25rem; /* px-5 py-2.5 */
        transition: all 0.2s ease-in-out;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .stButton > button:hover {
        opacity: 0.9;
    }

    /* Custom colors */
    .st-emotion-cache-1cypcdb { /* Target for main content padding */
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    .st-emotion-cache-1r6dm1x { /* Target for sidebar padding */
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Adjust Streamlit radio button appearance */
    .stRadio > label {
        display: flex;
        align-items: center;
        padding: 0.75rem;
        border: 1px solid #E5E7EB; /* border-gray-200 */
        border-radius: 0.5rem; /* rounded-lg */
        transition: all 0.2s ease-in-out;
        cursor: pointer;
    }
    .stRadio > label:hover {
        border-color: #10B981; /* hover:border-primary */
        background-color: rgba(16, 185, 129, 0.05); /* hover:bg-primary/5 */
    }
    .stRadio > label > div > p {
        margin-left: 0.75rem; /* ml-3 */
        color: #374151; /* text-gray-700 */
        font-weight: 500;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .stRadio [data-testid="stRadioInline"] > label > div:first-child {
        margin-top: 0 !important;
        margin-bottom: 0 !important;
    }

    /* Specific styles for info/success/warning boxes */
    .stAlert {
        border-radius: 0.5rem;
    }

    /* Custom styling for the header title and description */
    h1.st-emotion-cache-10q700h { /* Targeting the main title */
        font-size: 3rem; /* md:text-5xl */
        font-weight: 700; /* font-bold */
        color: #1F2937; /* text-dark */
        text-align: center;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 1rem;
        margin-bottom: 0.5rem;
    }
    h1.st-emotion-cache-10q700h .fas { /* Icon in title */
        color: #10B981; /* text-primary */
        font-size: 3.5rem; /* Adjusted for visual balance */
    }
    .st-emotion-cache-10q700h + div > p { /* Targeting the description below title */
        color: #4B5563; /* text-gray-600 */
        max-width: 56rem; /* max-w-2xl mx-auto */
        margin-left: auto;
        margin-right: auto;
        text-align: center;
        font-size: 1.125rem; /* text-lg */
    }

    /* Style for the confidence slider value display */
    .confidence-value-display {
        background-color: #10B981; /* bg-primary */
        color: white;
        font-weight: 500;
        padding: 0.25rem 0.5rem; /* px-2 py-1 */
        border-radius: 0.25rem; /* rounded */
        font-size: 0.875rem; /* text-sm */
    }

    /* Legend styling */
    .legend-item {
        display: flex;
        align-items: center;
        margin-bottom: 0.5rem;
    }
    .legend-color-box {
        width: 0.75rem; /* w-3 */
        height: 0.75rem; /* h-3 */
        border-radius: 9999px; /* rounded-full */
        margin-right: 0.5rem; /* mr-2 */
    }
    .legend-text {
        font-size: 0.875rem; /* text-sm */
    }

    /* Status card styling */
    .status-item {
        display: flex;
        align-items: center;
        margin-bottom: 0.75rem;
    }
    .status-indicator {
        width: 0.5rem; /* w-2 */
        height: 0.5rem; /* h-2 */
        border-radius: 9999px; /* rounded-full */
        margin-right: 0.75rem; /* mr-3 */
    }
    .status-text {
        color: #374151; /* text-gray-700 */
    }

    /* Placeholder for image/video upload areas */
    .placeholder-box {
        border: 2px dashed #D1D5DB; /* border-2 border-dashed border-gray-300 */
        border-radius: 0.75rem; /* rounded-xl */
        display: flex;
        align-items: center;
        justify-content: center;
        height: 18rem; /* h-72, adjust as needed */
        text-align: center;
        flex-direction: column;
        color: #9CA3AF; /* text-gray-400 */
    }
    .placeholder-box i {
        font-size: 2.5rem; /* text-4xl */
        margin-bottom: 0.5rem; /* mb-2 */
    }
    .placeholder-box p {
        color: #9CA3AF; /* text-gray-400 */
    }

    /* Detection stats cards */
    .stats-card {
        border-radius: 0.5rem; /* rounded-lg */
        padding: 1rem; /* p-4 */
        display: flex;
        align-items: center;
        gap: 0.75rem; /* gap-3 */
    }
    .stats-icon-box {
        width: 2.5rem; /* w-10 */
        height: 2.5rem; /* h-10 */
        border-radius: 9999px; /* rounded-full */
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .stats-icon-box i {
        color: white;
    }
    .stats-label {
        color: #4B5563; /* text-gray-600 */
        font-size: 0.875rem; /* text-sm */
    }
    .stats-value {
        font-weight: 700; /* font-bold */
        color: #1F2937; /* text-gray-800 */
        font-size: 1.25rem; /* text-xl */
    }

    /* Live Camera specific styles */
    .glow-border {
        box-shadow: 0 0 15px rgba(16, 185, 129, 0.3); /* primary color glow */
    }
    .video-placeholder {
        background-color: black;
        aspect-ratio: 16 / 9;
        display: flex;
        align-items: center;
        justify-content: center;
        position: relative;
        border-radius: 1rem; /* rounded-2xl */
        overflow: hidden;
    }
    .video-placeholder .overlay {
        position: absolute;
        width: 100%;
        height: 100%;
        opacity: 0.2;
        background: linear-gradient(to right, #4CAF50, #2196F3); /* from-green-400 to-blue-500 */
    }
    .video-placeholder .content {
        z-index: 10;
        text-align: center;
        color: #D1D5DB; /* text-gray-300 */
    }
    .video-placeholder .content i {
        font-size: 4rem; /* text-6xl */
    }

    /* Footer styles */
    .footer-badge {
        background-color: #F3F4F6; /* bg-gray-100 */
        border: 1px solid #E5E7EB; /* border-gray-200 */
        border-radius: 0.5rem; /* rounded-lg */
        padding: 0.5rem 1rem; /* py-2 px-4 */
        font-weight: 500; /* font-medium */
    }
    .footer-text {
        color: #4B5563; /* text-gray-600 */
    }
    .footer-highlight {
        color: #10B981; /* text-primary */
        font-weight: 500;
    }

    /* Animations */
    .pulse-animation {
        animation: pulse-stream 2s infinite;
    }
    @keyframes pulse-stream {
        0% { box-shadow: 0 0 0 0 rgba(59, 130, 246, 0.4); }
        70% { box-shadow: 0 0 0 10px rgba(59, 130, 246, 0); }
        100% { box-shadow: 0 0 0 0 rgba(59, 130, 246, 0); }
    }
</style>
"""

# Inject custom CSS
st.markdown(custom_css, unsafe_allow_html=True)

# --- Model Loading ---
@st.cache_resource
def load_yolo_model(url):
    """
    Downloads the YOLOv8 model from the provided URL and loads it.
    Uses Streamlit's caching to avoid re-downloading on every run.
    """
    try:
        st.info("Attempting to download model...")
        response = requests.get(url, stream=True) # Use stream=True for potentially large files
        response.raise_for_status()  # Raise an exception for bad status codes

        # Save the model to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as temp_model_file:
            for chunk in response.iter_content(chunk_size=8192):
                temp_model_file.write(chunk)
            model_path = temp_model_file.name

        st.success(f"Model downloaded to {model_path}. Loading YOLO model...")
        model = YOLO(model_path)
        st.success("YOLO model loaded successfully!")
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
        self.crop_count = 0
        self.weed_count = 0

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Perform inference
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

        self.crop_count = current_crop_count
        self.weed_count = current_weed_count
        return annotated_img

# --- UI and Main Logic ---
# URL to the raw model file on GitHub
MODEL_URL = "https://raw.githubusercontent.com/blurerjr/multidataset-weed-detection/master/best.pt"

# Load the model
model = None
try:
    with st.spinner("Downloading and loading the model (this may take a moment)..."):
        model = load_yolo_model(MODEL_URL)
except Exception as e:
    st.error(f"Failed to load model at startup: {e}")

if model is None:
    st.warning("Model could not be loaded. Please check the URL and your internet connection.")
    st.stop()

# Class names and colors
CLASS_NAMES = ['crop', 'weed']
COLORS = {
    'crop': (0, 255, 0), # Green
    'weed': (0, 0, 255)  # Red
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
col1, col2 = st.columns([1, 3]) # Mimic lg:grid-cols-4 layout

with col1: # Sidebar
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

    # Detection Mode
    st.markdown('<h3 style="font-weight: 500; color: #374151; margin-bottom: 0.5rem;">Detection Mode</h3>', unsafe_allow_html=True)
    detection_mode = st.radio(
        "Select Detection Mode",
        ["Image", "Video", "Live Camera"],
        key="detection_mode_radio",
        label_visibility="collapsed" # Hide default Streamlit label
    )

    st.markdown('<div style="margin-bottom: 1.5rem;"></div>', unsafe_allow_html=True) # Spacer

    # Confidence Threshold
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
        0.0, 1.0, 0.3, 0.05,
        key="confidence_slider",
        label_visibility="collapsed" # Hide default Streamlit label
    )
    # Update the JS-controlled span with the current slider value
    st.markdown(f"<script>document.getElementById('confidenceValue').textContent = '{confidence_threshold:.2f}';</script>", unsafe_allow_html=True)

    st.markdown('<div style="margin-bottom: 1.5rem;"></div>', unsafe_allow_html=True) # Spacer

    # Legend
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
    st.markdown('</div>', unsafe_allow_html=True) # Close detection-card

    # Status Card
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
    st.markdown('</div>', unsafe_allow_html=True) # Close detection-card


with col2: # Main Content
    # --- Image Detection Logic ---
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

        uploaded_file = st.file_uploader(
            "Upload Image",
            type=["jpg", "jpeg", "png"],
            key="image_uploader",
            label_visibility="collapsed" # Hide default Streamlit label
        )

        image_col, processed_image_col = st.columns(2)

        original_image_placeholder = image_col.empty()
        processed_image_placeholder = processed_image_col.empty()

        if uploaded_file is None:
            original_image_placeholder.markdown(
                """
                <h3 style="font-weight: 500; color: #374151; margin-bottom: 0.5rem; display: flex; align-items: center; gap: 0.5rem;">
                    <i class="fas fa-file-image" style="color:#6B7280;"></i> Original Image
                </h3>
                <div class="placeholder-box">
                    <i class="fas fa-image"></i>
                    <p>Upload an image to get started</p>
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
        else:
            image = Image.open(uploaded_file)
            img_array = np.array(image)

            original_image_placeholder.image(image, caption="Uploaded Image", use_container_width=True)
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


        if uploaded_file is not None and st.button("Detect Weeds & Crops", key="detect_image_button"):
            with st.spinner("Processing image..."):
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

                # Display detection stats
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
        else:
            st.markdown(
                """
                <div style="margin-top: 1.5rem; display: flex; align-items: center; gap: 1rem;">
                    <button class="stButton" style="background-color:#3B82F6; color:white;">
                        <i class="fas fa-search"></i> Detect Weeds & Crops
                    </button>
                    <div style="background-color:#FFFBEB; border:1px solid #FDE68A; border-radius:0.5rem; padding:0.5rem 1rem; color:#92400E; font-size:0.875rem; flex:1; display:flex; align-items:center; gap:0.25rem;">
                        <i class="fas fa-info-circle"></i> Upload an image and click detect to process.
                    </div>
                </div>
                """, unsafe_allow_html=True
            )
        st.markdown('</div>', unsafe_allow_html=True) # Close detection-card

    # --- Video Detection Logic ---
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
            # Save uploaded video to a temporary file
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

                    # Perform detection on the frame
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
        else:
            st.markdown(
                """
                <div style="margin-top: 1.5rem; display: flex; align-items: center; gap: 1rem;">
                    <button class="stButton" style="background-color:#3B82F6; color:white;">
                        <i class="fas fa-play-circle"></i> Process Video
                    </button>
                    <div style="background-color:#EFF6FF; border:1px solid #DBEAFE; border-radius:0.5rem; padding:0.5rem 1rem; color:#1E40AF; font-size:0.875rem; flex:1; display:flex; align-items:center; gap:0.25rem;">
                        <i class="fas fa-info-circle"></i> Processing may take time depending on video length.
                    </div>
                </div>
                """, unsafe_allow_html=True
            )
        st.markdown('</div>', unsafe_allow_html=True) # Close detection-card

    # --- Live Camera Detection Logic ---
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

        # Placeholder for the webrtc_streamer output
        webrtc_container = st.container()
        
        # Placeholders for live stats
        live_status_col, live_detection_col = st.columns(2)
        camera_status_placeholder = live_status_col.empty()
        active_detections_placeholder = live_detection_col.empty()

        # Initial status display
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

        # Use webrtc_streamer to get the live camera feed
        # We need to capture the context to access the transformer's counts
        webrtc_ctx = webrtc_streamer(
            key="weed_detection_live_stream",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration={
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
            },
            video_transformer_factory=lambda: WeedCropDetector(model, CLASS_NAMES, COLORS, confidence_threshold),
            media_stream_constraints={"video": True, "audio": False},
            async_transform=True,
            container_width=True # Ensure it takes available width
        )

        # Update live stats based on the transformer's internal counts
        if webrtc_ctx.video_transformer:
            # This loop runs continuously to update the UI with detection counts
            # It's not ideal for high-performance apps as it reruns Streamlit,
            # but it's the standard way to get data out of the transformer for display.
            while True:
                camera_status_placeholder.markdown(
                    """
                    <h3 style="font-weight: 500; color: #374151; margin-bottom: 0.5rem;">Camera Status</h3>
                    <div style="display: flex; align-items: center;">
                        <div class="status-indicator bg-green-500 pulse-animation"></div>
                        <span class="status-text">Active</span>
                    </div>
                    """, unsafe_allow_html=True
                )
                active_detections_placeholder.markdown(
                    f"""
                    <h3 style="font-weight: 500; color: #374151; margin-bottom: 0.5rem;">Active Detections</h3>
                    <div style="display:flex; gap:1rem;">
                        <div style="text-align:center;">
                            <p style="color:#4B5563; font-size:0.875rem;">Crops</p>
                            <p style="font-weight:bold; color:#1F2937; font-size:1.25rem;">{webrtc_ctx.video_transformer.crop_count}</p>
                        </div>
                        <div style="text-align:center;">
                            <p style="color:#4B5563; font-size:0.875rem;">Weeds</p>
                            <p style="font-weight:bold; color:#1F2937; font-size:1.25rem;">{webrtc_ctx.video_transformer.weed_count}</p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True
                )
                time.sleep(0.5) # Update every 0.5 seconds to reduce reruns

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
        st.markdown('</div>', unsafe_allow_html=True) # Close detection-card

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
            Developed by <span class="footer-highlight">US</span> using Ultralytics YOLO v8 and Streamlit.
        </p>
    </footer>
    """,
    unsafe_allow_html=True
)
