from PIL import Image
import pytesseract
import os

# HARD-CODE TESSERACT PATH
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

if not os.path.exists(TESSERACT_PATH):
    # Fallback to check Program Files (x86) just in case
    ALT_PATH = r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"
    if os.path.exists(ALT_PATH):
        TESSERACT_PATH = ALT_PATH
    else:
        # We'll log it but not crash immediately to allow the error dict return
        print(f"CRITICAL: Tesseract not found at {TESSERACT_PATH}")

pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

def extract_text_from_image(image_path: str) -> dict:
    """
    Extract text from image using Tesseract OCR with hardcoded pathing for Windows.
    """
    try:
        if not os.path.exists(TESSERACT_PATH):
            return {
                "extracted_text": "Error: Tesseract OCR binary not found. Please ensure it is installed at C:\\Program Files\\Tesseract-OCR",
                "confidence": 0.0,
                "error": "Binary missing"
            }

        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)

        confidence = 0.85 if len(text.strip()) > 10 else 0.4

        return {
            "extracted_text": text.strip(),
            "confidence": confidence
        }
    except Exception as e:
        return {
            "extracted_text": "",
            "confidence": 0.0,
            "error": str(e)
        }
