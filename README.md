# Passport & Document OCR API

This is a Flask-based document parsing API that performs OCR and structured field extraction from scanned identity documents like passports, PAN, Aadhaar, and more.

It uses *PaddleOCR* , *Tesseract* , etc  under the hood with custom logic for smart field parsing, image preprocessing, and orientation correction.

---

##  Features

-  Automatic text extraction from PDFs or images
-  Orientation correction using OSD
-  Image enhancement (contrast, sharpening, upscaling)
-  Smart field parsing 
-  PDF-to-Image conversion
-  API-based architecture (ready for LangChain Q&A)

---

##  Technologies Used

- Python, Flask
- OpenCV
- PaddleOCR
- Pytesseract
- NumPy

---

##  How to Run Locally

```bash
git clone https://github.com/OmMathur23/ocr-.git
cd ocr-

# Set up a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
