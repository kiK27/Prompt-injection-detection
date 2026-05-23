import streamlit as st
from detector.text_detector import TextDetector
from detector.semantic_detector import SemanticDetector
from detector.image_detector import ImageDetector
from detector.risk_scorer import RiskScorer
from detector.utils import ensure_temp_exists

ensure_temp_exists()

st.set_page_config(
    page_title="Prompt Injection Detector",
    layout="wide"
)

st.title("AI Prompt Injection Detector")

text_detector = TextDetector()
semantic_detector = SemanticDetector()
image_detector = ImageDetector()
risk_scorer = RiskScorer()

input_text = st.text_area(
    "Enter Prompt",
    height=250
)

uploaded_image = st.file_uploader(
    "Upload Image",
    type=["png", "jpg", "jpeg"]
)

if st.button("Analyze"):

    if input_text:

        text_result = text_detector.detect(input_text)

        semantic_result = semantic_detector.detect(input_text)

        risk_result = risk_scorer.calculate(
            text_result["keyword_score"],
            semantic_result["semantic_score"]
        )

        st.subheader("Detection Results")

        st.write("### Semantic Classification")
        st.write(semantic_result["classification"])

        st.write("### Risk Score")
        st.write(risk_result["risk_score"])

        st.write("### Risk Level")
        st.write(risk_result["risk_level"])

        st.write("### Keyword Matches")
        st.write(text_result["matches"])

    if uploaded_image:

        image_path = f"temp/{uploaded_image.name}"

        with open(image_path, "wb") as f:
            f.write(uploaded_image.getbuffer())

        image_result = image_detector.detect(image_path)

        st.subheader("Image Analysis")

        st.write("### Extracted Text")
        st.text(image_result["ocr_text"])

        st.write("### Suspicious Matches")
        st.write(image_result["matches"])