import streamlit as st
import cv2
import tempfile
import numpy as np
from ultralytics import YOLO

# ---------------- UI ----------------
st.set_page_config(page_title="AI Distraction Detection", layout="wide")

st.title("🎯 AI Distraction Detection System")
st.markdown("Detects student focus using AI (Phone usage, presence, attention)")

# Sidebar
st.sidebar.header("Controls")
confidence_threshold = st.sidebar.slider("Detection Confidence", 0.1, 1.0, 0.5)

# Load model
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# ---------------- Video Upload ----------------
video_file = st.file_uploader("📂 Upload a video", type=["mp4", "avi", "mov"])

# Stats
focus_score = 100
focus_scores = []
distraction_time = 0
total_frames = 0

# Layout
col1, col2 = st.columns([2, 1])
frame_window = col1.empty()
stats_box = col2.empty()

# ---------------- Processing ----------------
if video_file is not None:

    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())

    cap = cv2.VideoCapture(tfile.name)

    st.success("✅ Processing video...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        total_frames += 1

        results = model(frame, conf=confidence_threshold)
        annotated_frame = results[0].plot()

        labels = results[0].names
        detected = [labels[int(cls)] for cls in results[0].boxes.cls]

        # ---------------- Logic ----------------
        status = "Focused"

        if "cell phone" in detected:
            distraction_time += 1
            status = "Using Phone"
            focus_score -= 0.5

        elif "person" not in detected:
            distraction_time += 1
            status = "No Person Detected"
            focus_score -= 0.7

        # Keep score in range
        focus_score = max(0, min(100, focus_score))
        focus_scores.append(focus_score)

        # ---------------- Display ----------------
        frame_window.image(annotated_frame, channels="BGR")

        with stats_box.container():
            st.metric("📊 Focus Score", f"{int(focus_score)}%")
            st.write(f"🧠 Status: **{status}**")
            st.write(f"⏱️ Distraction Frames: {distraction_time}")

            if focus_score < 50:
                st.warning("⚠️ You are distracted!")

    cap.release()

    # ---------------- Graph ----------------
    st.subheader("📈 Focus Score Over Time")
    st.line_chart(focus_scores)

    # ---------------- Summary ----------------
    st.subheader("📋 Final Report")

    distraction_percentage = (distraction_time / total_frames) * 100 if total_frames > 0 else 0

    st.write(f"Total Frames: {total_frames}")
    st.write(f"Distraction Time: {distraction_time}")
    st.write(f"Distraction %: {distraction_percentage:.2f}%")
    st.write(f"Final Focus Score: {int(focus_score)}%")

else:
    st.info("👆 Upload a video to start detection")
