import streamlit as st
import sqlite3
import cv2
import cvzone
import time
import pickle
from collections import defaultdict
import pandas as pd
from ultralytics import YOLO
import av
import numpy as np
import os
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# Files to persist detection data
COUNTS_FILE = "final_counts.pkl"
FRAMES_FILE = "final_frames.pkl"

#########################################
#         PAGE CONFIGURATION & STYLE    #
#########################################
st.set_page_config(page_title="Live Member Face Recognition")
st.markdown(
    """
    <style>
    body { background-color: #001F3D; }
    h1, h2, h3 { color: #2ECC71 !important; }
    .sidebar .stTitle { color: #2ECC71 !important; }
    .sidebar .stSlider, .sidebar .stButton {
        background-color: #003366 !important;
        color: white !important;
        font-weight: bold;
        border-radius: 10px;
    }
    .sidebar .stButton:hover { background-color: #2ECC71; }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown("<h1 class='stTitle'>Live Member Face Recognition</h1>", unsafe_allow_html=True)
st.markdown("This app performs live face recognition using YOLO and overlays member details from a SQLite database.")

#########################################
#           FIELD NAMES (DEFINE FIRST)  #
#########################################
FIELD_NAMES = [
    "ID", "Name", "Age", "Date of Birth", "Address", "Loyalty",
    "Member Since", "Gender", "Email", "Phone Number", "Membership Type",
    "Status", "Occupation", "Interests", "Marital Status"
]

#########################################
#            SIDEBAR SETTINGS           #
#########################################
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.3, 0.01)
font_threshold = st.sidebar.slider("Font Scale", 0.0, 5.0, 0.7, 0.1)  # Increased font scale for visibility
thickness_threshold = st.sidebar.slider("Text Thickness", 0, 10, 1, 1)  # Increased thickness for better readability
line_threshold = st.sidebar.slider("Line Height (px)", 0, 100, 30, 1)

# Dropdown to select fields to show
selected_fields = st.sidebar.multiselect(
    "Select Fields to Display",
    options=FIELD_NAMES,  # All 15 fields
    default=FIELD_NAMES  # Default to all fields selected
)

# Input Method for live webcam or video file
option = st.selectbox(
    "How would you like to make Predictions?",
    ("Web Cam", "Video File"),
    index=0,
    help="Select 'Web Cam' for a live feed or 'Video File' to upload a video."
)

#########################################
#           SQLITE HELPER               #
#########################################
def getProfile(member_id):
    conn = sqlite3.connect('capstone.db')
    cmd = "SELECT * FROM ski WHERE id=?"
    cursor = conn.execute(cmd, (member_id,))
    profile = cursor.fetchone()
    conn.close()
    return profile

#########################################
#           LOAD YOLO MODEL             #
#########################################
model = YOLO("best.pt")
classNames = ['angela', 'classmate', 'giuliana', 'javier', 'john',
              'maite', 'mike', 'ron', 'shanti', 'tom', 'vilma', 'will',"kevin","shirley"]

#########################################
#        YOLO WEBCAM PROCESSING         #
#########################################
class YOLOVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.prev_frame_time = time.time()
        self.detection_counts = defaultdict(int)
        self.detected_frames = {}

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        new_frame_time = time.time()
        detections_this_frame = []
        results = model(img, conf=confidence_threshold, stream=True)
        highest_conf_info = None
        
        for r in results:
            for box in r.boxes:
                conf = float(box.conf[0])
                if conf < confidence_threshold:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                profile = getProfile(cls + 1)  # assume DB IDs are 1-indexed
                if profile is not None:
                    detected_name = profile[1]
                    detected_status = profile[11] if len(profile) > 11 else "Inactive"
                else:
                    detected_name = classNames[cls] if 0 <= cls < len(classNames) else "Unknown"
                    detected_status = "Inactive"
                
                self.detection_counts[detected_name] += 1
                conf_percent = int(conf * 100)
                
                # Set bounding box color based on status (green for Active, red for Inactive)
                background_color = (0, 255, 0) if detected_status.lower() == "active" else (0, 0, 255)
                
                # Draw bounding box with dynamic color
                cv2.rectangle(img, (x1, y1), (x2, y2), background_color, thickness_threshold)
                
                # Text color is always black for the name-status-confidence
                label_text = f'{detected_name} - {detected_status} {conf_percent}%'
                text_color = (0, 255, 0)  # Text color is black
                cvzone.putTextRect(img, label_text, (max(0, x1), max(35, y1)),
                                   scale=font_threshold * 3, 
                                   thickness=thickness_threshold,colorT=(0, 0, 0),
                                   colorR=background_color)  # Text color is black

                if conf > (highest_conf_info['confidence'] if highest_conf_info else 0):
                    highest_conf_info = {'name': detected_name, 'status': detected_status, 'confidence': conf}
                
                # Show extended data if profile
                if profile is not None:
                    h = y2 - y1
                    startY = y1 + h + 20
                    for i, field_name in enumerate(FIELD_NAMES):
                        if field_name in selected_fields:  # Display only the selected fields
                            if i < len(profile):
                                text = f"{field_name}: {profile[i]}"
                                (text_w, text_h), _ = cv2.getTextSize(
                                    text, cv2.FONT_HERSHEY_SIMPLEX, font_threshold, thickness_threshold
                                )
                                cv2.rectangle(
                                    img,
                                    (x1, startY + i * line_threshold - text_h - 5),
                                    (x1 + text_w, startY + i * line_threshold + 5),
                                    (0, 0, 0), cv2.FILLED
                                )
                                cv2.putText(
                                    img,
                                    text,
                                    (x1, startY + i * line_threshold),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    font_threshold,
                                    (0, 255, 0),
                                    thickness_threshold
                                )
                
                detections_this_frame.append((detected_name, x1, y1, x2, y2, profile))
        
        if highest_conf_info is not None:
            global_text = f"{highest_conf_info['name']} - {highest_conf_info['status']}"
            global_color = (0, 255, 0) if highest_conf_info['status'].lower() == "active" else (0, 0, 255)
            cv2.putText(img, global_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        font_threshold * 2.5, global_color, 3)

        for (det_name, x1, y1, x2, y2, profile) in detections_this_frame:
            self.detected_frames[det_name] = (img.copy(), profile)

        with open(COUNTS_FILE, "wb") as f:
            pickle.dump(dict(self.detection_counts), f)
        with open(FRAMES_FILE, "wb") as f:
            pickle.dump(self.detected_frames, f)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# Start webcam stream only if "Web Cam" mode is selected.
if option == "Web Cam":
    ctx = webrtc_streamer(
        key="webcam", 
        video_processor_factory=YOLOVideoProcessor,
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        },
        media_stream_constraints={
            "video": {
                "width": {"ideal": 1280},    # ✅ Force Full HD 1080p
                "height": {"ideal": 720},   # ✅ Full 1080p height
                "frameRate": {"ideal": 30},  # ✅ Force buttery smooth 60 FPS
                "aspectRatio": 1.777778,     # ✅ Maintain 16:9 ratio
                "facingMode": "user"         # ✅ Force front camera
            }
        },
        async_processing=True
    )
else:
    ctx = None


# When stream stops (only for Web Cam mode), show results.
if option == "Web Cam" and ctx is not None and ctx.state is not None and not ctx.state.playing:
    st.write("Webcam stream stopped.")
    if os.path.exists(COUNTS_FILE) and os.path.exists(FRAMES_FILE):
        with open(COUNTS_FILE, "rb") as f:
            final_counts = pickle.load(f)
        with open(FRAMES_FILE, "rb") as f:
            final_frames = pickle.load(f)
    else:
        final_counts = {}
        final_frames = {}

    if st.button("Show Results"):
        if final_counts:
            df = pd.DataFrame(list(final_counts.items()), columns=["Name", "Detections"])
            df["Confidence %"] = round(confidence_threshold * 100, 2)
            df["Status"] = df["Name"].apply(lambda x: "Active" if x.lower() in [n.lower() for n in classNames] else "Inactive")
            st.table(df)
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Results", data=csv, file_name="detection_results.csv", mime="text/csv")
        else:
            st.write("No detections to display.")

        if final_counts:
            max_name = max(final_counts, key=final_counts.get)
            max_count = final_counts[max_name]
            st.write(f"**Highest Detection: {max_name} ({max_count} times)**")
            if max_name in final_frames:
                full_img, profile = final_frames[max_name]
                desired_width = 800
                scale = desired_width / full_img.shape[1]
                resized_img = cv2.resize(full_img, (desired_width, int(full_img.shape[0] * scale)))
                overlay_height = 150
                overlay = np.zeros((overlay_height, desired_width, 3), dtype=np.uint8)
                if profile is not None:
                    lines = []
                    for i, field in enumerate(FIELD_NAMES):
                        if i < len(profile):
                            lines.append(f"{field}: {profile[i]}")
                    y0 = 30
                    dy = 25
                    for idx, line in enumerate(lines):
                        cv2.putText(overlay, line, (10, y0 + idx * dy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                final_img = np.vstack((resized_img, overlay))
                st.markdown("### Final Annotated Image (Highest Detection)")
                st.image(final_img, channels="BGR", use_column_width=True)
                cv2.imwrite(f"{max_name}_final_full.jpg", final_img)
            else:
                st.write("No captured image for the highest detection.")
        else:
            st.write("No captured images to display.")


#         VIDEO FILE PROCESSING         #
#########################################
if option == "Video File":
    st.write("### Video File Processing")
    uploaded_video = st.file_uploader("Upload Video", type=["mp4", "mov", "avi"], key="video_uploader")
    rotate_mode = st.selectbox("Rotate Video", 
                               ["None", "90° CW", "180°", "90° CCW", "Flip Horizontal"],
                               key="rotate_video")
    adjust = st.slider("Adjust Scale", 0.1, 2.0, 1.0, 0.1, key="adjust_scale")
    out_width = st.number_input("Output Width", min_value=100, value=640, key="out_width")
    out_height = st.number_input("Output Height", min_value=100, value=480, key="out_height")
    process_button = st.button("Process Video", key="process_video")
    
    def process_video_file(
        input_bytes, rotate_mode, adjust, output_size,
        conf_threshold, font_threshold, thickness_threshold, line_threshold
    ):
        """
        Process the video using all user-chosen parameters.
        """
        input_path = "temp_input_video.mp4"
        output_path = "temp_output_video.mp4"
        with open(input_path, "wb") as f:
            f.write(input_bytes.read())
        
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            st.error("Error opening video file.")
            return None
        
        orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        st.info(f"Original video size (width x height): {orig_width} x {orig_height}, FPS: {fps}")
        
        out_w, out_h = output_size
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))
        
        detection_counts_vid = defaultdict(int)
        detected_frames_vid = {}
        preview_placeholder = st.empty()
        prev_frame_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if rotate_mode == "90° CW":
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            elif rotate_mode == "180°":
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            elif rotate_mode == "90° CCW":
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            elif rotate_mode == "Flip Horizontal":
                frame = cv2.flip(frame, 1)
            
            new_frame_time = time.time()
            highest_conf_value = 0
            highest_conf_info = None
            detections_this_frame = []
            
            results = model(frame, conf=conf_threshold, stream=True)
            for r in results:
                for box in r.boxes:
                    conf = float(box.conf[0])
                    if conf < conf_threshold:
                        continue
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls = int(box.cls[0])
                    profile = getProfile(cls + 1)
                    
                    if profile is not None:
                        detected_name = profile[1]
                        detected_status = profile[11] if len(profile) > 11 else "Inactive"
                    else:
                        detected_name = classNames[cls] if 0 <= cls < len(classNames) else "Unknown"
                        detected_status = "Inactive"
                    
                    detection_counts_vid[detected_name] += 1
                    conf_percent = int(conf * 100)
                    label_text = f"{detected_name} - {detected_status} {conf_percent}%"
                    box_color = (0, 255, 0) if detected_status.lower() == "active" else (0, 0, 255)
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, thickness_threshold)
                    cvzone.putTextRect(frame, label_text, (max(0, x1), max(35, y1)),
                                       scale=font_threshold * 2,
                                       thickness=thickness_threshold,
                                       colorR=box_color)
                    
                    if conf > highest_conf_value:
                        highest_conf_value = conf
                        highest_conf_info = {'name': detected_name, 'status': detected_status}
                    
                    if profile is not None:
                        h = y2 - y1
                        startY = y1 + h + 20
                        for i, field_name in enumerate(FIELD_NAMES):
                            if i < len(profile):
                                t = f"{field_name}: {profile[i]}"
                                (text_w, text_h), _ = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX,
                                                                      font_threshold, thickness_threshold)
                                cv2.rectangle(
                                    frame,
                                    (x1, startY + i * line_threshold - text_h - 5),
                                    (x1 + text_w, startY + i * line_threshold + 5),
                                    (0, 0, 0), cv2.FILLED
                                )
                                cv2.putText(
                                    frame,
                                    t,
                                    (x1, startY + i * line_threshold),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    font_threshold,
                                    (0, 255, 0),
                                    thickness_threshold
                                )
                    detections_this_frame.append((detected_name, x1, y1, x2, y2))
            
            if highest_conf_info is not None:
                global_text = f"{highest_conf_info['name']} - {highest_conf_info['status']}"
                global_color = (0, 255, 0) if highest_conf_info['status'].lower() == "active" else (0, 0, 255)
                cv2.putText(frame, global_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                            font_threshold * 2.5, global_color, 3)
            
            for (det_name, x1, y1, x2, y2) in detections_this_frame:
                face_crop = frame[y1:y2, x1:x2].copy()
                detected_frames_vid[det_name] = (frame.copy(), face_crop)
            
            out_frame = cv2.resize(frame, (out_w, out_h))
            out.write(out_frame)
            
            fps_val = 1 / (new_frame_time - prev_frame_time) if prev_frame_time else 0
            prev_frame_time = new_frame_time
            fps_text = f"FPS: {fps_val:.2f}"
            (tw, th), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            pos = (frame.shape[1] - tw - 10, th + 10)
            cv2.putText(frame, fps_text, pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)
            
            preview_frame = cv2.resize(frame, (0, 0), fx=adjust, fy=adjust) if adjust != 1.0 else frame
            preview_placeholder.image(preview_frame, channels="BGR")
        
        cap.release()
        out.release()
        preview_placeholder.empty()
        
        st.success("Video processing complete!")
        with open(output_path, "rb") as vid_file:
            st.download_button(
                label="Download Processed Video",
                data=vid_file,
                file_name=output_path,
                mime="video/mp4"
            )
        
        if detection_counts_vid:
            df_vid = pd.DataFrame(list(detection_counts_vid.items()), columns=["Name", "Detections"])
            df_vid["Confidence %"] = round(conf_threshold * 100, 2)
            df_vid["Status"] = df_vid["Name"].apply(
                lambda x: "Active" if x.lower() in [n.lower() for n in classNames] else "Inactive"
            )
            st.table(df_vid)
        
        return detection_counts_vid, detected_frames_vid

    if uploaded_video is not None and process_button:
        detection_counts_vid, detected_frames_vid = process_video_file(
            uploaded_video,
            rotate_mode,
            adjust,
            (int(out_width), int(out_height)),
            confidence_threshold,
            font_threshold,
            thickness_threshold,
            line_threshold
        )
