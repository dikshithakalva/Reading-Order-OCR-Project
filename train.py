import json
import numpy as np
import streamlit as st
import cv2
import pytesseract
import time
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def extract_unordered_bounding_boxes(image):
    """Extract bounding boxes from image using Tesseract OCR."""
    start_time = time.time()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    boxes = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
    unordered_bboxes = []
    
    for i in range(len(boxes['text'])):
        if boxes['text'][i].strip():
            x, y, w, h = boxes['left'][i], boxes['top'][i], boxes['width'][i], boxes['height'][i]
            unordered_bboxes.append({
                "id": i + 1,
                "bounding_box": {"x_min": x, "y_min": y, "x_max": x + w, "y_max": y + h},
                "text": boxes['text'][i]
            })
    
    processing_time = round(time.time() - start_time, 4)
    return unordered_bboxes, processing_time

def order_bounding_boxes(unordered_bboxes):
    """Sort bounding boxes in reading order."""
    start_time = time.time()
    if not isinstance(unordered_bboxes, list):
        print("Error: Invalid bounding box format! Expected a list.")
        return [], 0
    
    ordered_bboxes = sorted(
        unordered_bboxes, key=lambda b: (b['bounding_box']['y_min'], b['bounding_box']['x_min'])
    )
    
    processing_time = round(time.time() - start_time, 4)
    return ordered_bboxes, processing_time

def draw_bounding_boxes(image, bboxes):
    """Draw bounding boxes on the image."""
    for bbox in bboxes:
        x_min, y_min, x_max, y_max = bbox['bounding_box'].values()
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(image, bbox['text'], (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return image

def main():
    st.title("Reading Order Detection")
    option = st.radio("Choose Input Type", ["Upload Image", "Upload Unordered JSON"])
    
    if option == "Upload Image":
        uploaded_image = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])
        if uploaded_image is not None:
            file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            unordered_bboxes, extraction_time = extract_unordered_bounding_boxes(image)
            ordered_bboxes, sorting_time = order_bounding_boxes(unordered_bboxes)
            
            st.success("Bounding box extraction and ordering completed!")
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original Image")
                st.image(image, channels="BGR")
            
            with col2:
                st.subheader("Processed Image with Bounding Boxes")
                processed_image = draw_bounding_boxes(image.copy(), ordered_bboxes)
                st.image(processed_image, channels="BGR")
                
            st.download_button("Download Processed Image", data=cv2.imencode('.png', processed_image)[1].tobytes(), file_name="processed_image.png", mime="image/png")
            st.download_button("Download Ordered JSON", data=json.dumps(ordered_bboxes, indent=4), file_name="ordered_bboxes.json", mime="application/json")

    elif option == "Upload Unordered JSON":
        uploaded_json = st.file_uploader("Upload an unordered JSON file", type=["json"])
        if uploaded_json is not None:
            bboxes = json.load(uploaded_json)
            if "words" in bboxes:
                bboxes = bboxes["words"]
            
            ordered_bboxes, sorting_time = order_bounding_boxes(bboxes)
            
            st.success("Bounding box ordering completed!")
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Unordered Bounding Boxes")
                st.json(bboxes)
            with col2:
                st.subheader("Ordered Bounding Boxes")
                st.json(ordered_bboxes)
            
            st.download_button("Download Ordered JSON", data=json.dumps(ordered_bboxes, indent=4), file_name="ordered_bboxes.json", mime="application/json")

if __name__ == "__main__":
    main()