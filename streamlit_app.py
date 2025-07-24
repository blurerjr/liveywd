import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import requests
from PIL import Image
import os
# Import necessary components for live camera streaming
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode

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
        # Using tempfile.NamedTemporaryFile is safer for cleanup
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

    def transform(self, frame):
        # Convert the WebRTC frame (av.VideoFrame) to an OpenCV numpy array (BGR format)
        img = frame.to_ndarray(format="bgr24")

        # Perform inference with the YOLO model
        # stream=False for single image inference, verbose=False to suppress console output per frame
        results = self.model.predict(img, conf=self.confidence_threshold, verbose=False)

        # Draw bounding boxes and labels on the frame
        annotated_img = img.copy()
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0]
                cls_id = int(box.cls[0])

                # Ensure class_id is within the bounds of CLASS_NAMES
                if cls_id < len(self.class_names):
                    class_name = self.class_names[cls_id]
                else:
                    class_name = f"Unknown {cls_id}" # Fallback for unknown classes

                label = f"{class_name} {conf:.2f}"
                color = self.colors.get(class_name, (255, 255, 255)) # Default to white if class not in COLORS

                # Draw rectangle
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)

                # Put label background
                (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(annotated_img, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
                cv2.putText(annotated_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2) # Text in black

        return annotated_img # Return the processed frame

# --- UI and Main Logic ---
st.set_page_config(page_title="Weed & Crop Detection", layout="wide")
st.title("🌿 Weed & Crop Detection using YOLOv8")
st.write("Upload an image or a video, or use your live camera to detect weeds and crops. "
         "The YOLOv8 model is trained to identify these two classes.")

# URL to the raw model file on GitHub
MODEL_URL = "https://raw.githubusercontent.com/blurerjr/multidataset-weed-detection/master/best.pt"

# Load the model
model = None # Initialize model to None
try:
    with st.spinner("Downloading and loading the model (this may take a moment)..."):
        model = load_yolo_model(MODEL_URL)
except Exception as e:
    st.error(f"Failed to load model at startup: {e}")

if model is None:
    st.warning("Model could not be loaded. Please check the URL and your internet connection.")
    st.stop() # Stop the app if model loading failed

# Class names - assuming these are the classes the model was trained on.
# You can get the exact class names from the model's .yaml file if available.
CLASS_NAMES = ['crop', 'weed']
# Colors for bounding boxes (Crop: Green, Weed: Red)
COLORS = {
    'crop': (0, 255, 0), # Green
    'weed': (0, 0, 255)  # Red
}

# Sidebar for options
st.sidebar.title("Options")
detection_mode = st.sidebar.radio("Select Detection Mode", ["Image", "Video", "Live Camera"])
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.3, 0.05)

# --- Image Detection Logic ---
if detection_mode == "Image":
    st.header("Image Detection")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read the image
        image = Image.open(uploaded_file)
        img_array = np.array(image)

        st.image(image, caption="Uploaded Image", use_container_width=True)

        if st.button("Detect Weeds & Crops"):
            with st.spinner("Processing image..."):
                # Perform detection
                results = model.predict(img_array, conf=confidence_threshold)

                annotated_img_array = img_array.copy()

                detection_count = {'crop': 0, 'weed': 0}

                # Draw bounding boxes on the image
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
                        color = COLORS.get(class_name, (255, 255, 255)) # Default to white

                        # Draw rectangle
                        cv2.rectangle(annotated_img_array, (x1, y1), (x2, y2), color, 2)

                        # Put label background
                        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        cv2.rectangle(annotated_img_array, (x1, y1 - h - 5), (x1 + w, y1), color, -1)
                        cv2.putText(annotated_img_array, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

                st.image(annotated_img_array, caption="Processed Image", use_container_width=True)
                st.success(f"Detection complete! Found {detection_count['crop']} crops and {detection_count['weed']} weeds.")

# --- Video Detection Logic ---
elif detection_mode == "Video":
    st.header("Video Detection")
    uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "mov", "avi"])

    if uploaded_file is not None:
        # Save uploaded video to a temporary file
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        video_path = tfile.name
        tfile.close() # Close the file handle after writing

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            st.error("Error: Could not open video file.")
            os.remove(video_path) # Clean up temp file
        else:
            st.info("Processing video... This may take a while depending on video length.")
            stframe = st.empty() # Placeholder for the video frames

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Perform detection on the frame
                results = model.predict(frame, conf=confidence_threshold, verbose=False)

                annotated_frame = frame.copy()

                # Draw bounding boxes
                for result in results:
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = box.conf[0]
                        cls_id = int(box.cls[0])

                        if cls_id < len(CLASS_NAMES):
                            class_name = CLASS_NAMES[cls_id]
                        else:
                            class_name = f"Unknown {cls_id}"

                        label = f"{class_name} {conf:.2f}"
                        color = COLORS.get(class_name, (255, 255, 255))

                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        cv2.rectangle(annotated_frame, (x1, y1 - h - 5), (x1 + w, y1), color, -1)
                        cv2.putText(annotated_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

                # Display the annotated frame
                stframe.image(annotated_frame, channels="BGR", use_container_width=True)

            cap.release()
            os.remove(video_path) # Clean up the temp file
            st.success("Video processing complete.")

# --- Live Camera Detection Logic ---
elif detection_mode == "Live Camera":
    st.header("Live Camera Detection")
    st.info("Click 'Start' and grant camera access to begin real-time weed and crop detection.")

    # Use webrtc_streamer to get the live camera feed
    webrtc_streamer(
        key="weed_detection_live_stream", # Unique key for this component
        mode=WebRtcMode.SENDRECV, # Send video from browser and receive processed video
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}] # Public STUN server for NAT traversal
        },
        # Pass the model, class_names, colors, and confidence_threshold to the transformer
        video_transformer_factory=lambda: WeedCropDetector(model, CLASS_NAMES, COLORS, confidence_threshold),
        media_stream_constraints={"video": True, "audio": False}, # Request video only, no audio
        async_transform=True, # Process frames asynchronously to prevent UI blocking
    )

st.markdown("---")
st.markdown("Developed by blurerjr/mu using Ultralytics YOLO and Streamlit.")

