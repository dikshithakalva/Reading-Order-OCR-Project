# Reading-Order-OCR-Project

## 📄 Overview
This project presents a **layout-aware multilingual reading order detection system** designed for scanned documents. The system accurately reconstructs the natural human reading flow by detecting and reordering text blocks across documents written in **Telugu, Tamil, Hindi, and English**.

It integrates **Tesseract OCR**, document layout analysis, and **adaptive bounding-box sorting** to significantly improve structured text extraction from complex, multi-column scanned pages.

## 🚀 Key Features
- 🌐 Multilingual support: Telugu, Tamil, Hindi, English
- 📑 Layout-aware reading order detection
- 🔍 OCR-based text extraction using Tesseract
- 📦 Adaptive bounding-box sorting for complex layouts
- ⚙️ End-to-end Python-based processing pipeline
- 📊 Processes 1,000+ text blocks per document efficiently

---

## System Pipeline
1. **Preprocessing**
   - Image normalization and enhancement
   - Noise reduction for improved OCR accuracy

2. **Text Detection & OCR**
   - Multilingual text extraction using Tesseract
   - Bounding-box generation for detected text regions

3. **Layout Analysis**
   - Spatial clustering of bounding boxes
   - Column and line structure detection

4. **Reading Order Detection**
   - Adaptive sorting based on spatial coordinates
   - Language-agnostic layout reasoning

5. **Post-processing**
   - Structured JSON output
   - Reordered text blocks for downstream applications

![WhatsApp Image 2026-01-04 at 8 19 36 PM](https://github.com/user-attachments/assets/16c16fe0-5c33-45c7-abe5-5aae176a82d2)

<img width="751" height="482" alt="image" src="https://github.com/user-attachments/assets/d4028b38-4e2f-4def-b76b-6cbfcb75e769" />



