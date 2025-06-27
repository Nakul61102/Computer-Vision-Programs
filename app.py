import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import cv2
import numpy as np
from util import get_limits
import av


st.set_page_config(page_title="Color Detector", layout="centered")

st.title("🎨 Real-Time Color Detection with OpenCV")

bgr_color = st.color_picker("Pick a color to detect", "#FFFF00")
bgr_color = tuple(int(bgr_color.lstrip("#")[i:i+2], 16) for i in (4, 2, 0))

st.write(f"Detecting BGR Color: {bgr_color}")

lower, upper = get_limits(bgr_color)

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

class VideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        try:
            img = frame.to_ndarray(format="bgr24")
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, lower, upper)
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                if cv2.contourArea(cnt) > 300:
                    x, y, w, h = cv2.boundingRect(cnt)
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(img, "Color Detected", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # ✅ Return a VideoFrame (not raw image)
            return av.VideoFrame.from_ndarray(img, format="bgr24")
        except Exception as e:
            print("Error in recv():", e)
            return frame


webrtc_streamer(
    key="color-detect-v2",
    video_processor_factory=VideoProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True  # ✅ smoother processing
)


