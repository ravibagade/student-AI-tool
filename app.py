# =====================================================
# INSTALL REQUIRED LIBRARIES
# =====================================================
# pip install opencv-python pytesseract pillow google-generativeai
# =====================================================
# IMPORTS
# =====================================================
import cv2
import pytesseract
from PIL import Image
import google.generativeai as genai

print("Installed Successfully")

# =====================================================
# GEMINI API KEY (HARDCODED FOR NOW)
# =====================================================
api_key = "AIzaSyAG54BbA65C6lUOVkA5fPqmXdTksD9zyhk"
genai.configure(api_key=api_key)

# =====================================================
# TESSERACT PATH
# =====================================================
pytesseract.pytesseract.tesseract_cmd = \
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# =====================================================
# IMAGE PATH
# =====================================================
image_path = r"E:/New folder/21.jpg"

# =====================================================
# READ IMAGE
# =====================================================
img = cv2.imread(image_path)
if img is None:
    print("ERROR: Image not found!")
    exit()

# =====================================================
# IMAGE PREPROCESSING
# =====================================================
# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Thresholding
thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)[1]

# =====================================================
# SAVE PROCESSED IMAGE
# =====================================================
processed_path = "processed.png"
cv2.imwrite(processed_path, thresh)

# =====================================================
# OCR TEXT EXTRACTION
# =====================================================
text = pytesseract.image_to_string(Image.open(processed_path))

# =====================================================
# PRINT EXTRACTED TEXT
# =====================================================
print("\n=======================")
print("EXTRACTED TEXT")
print("=======================\n")
print(text)

# =====================================================
# PROMPT
# =====================================================
prompt = f"""
You are an expert MCQ solver.
Analyze the following MCQ carefully.
Return:
1. Correct Option
2. Correct Answer
3. Short Explanation

MCQ:
{text}
"""

# =====================================================
# GEMINI RESPONSE
# =====================================================
model = genai.GenerativeModel("gemini-2.0-flash")
response = model.generate_content(prompt)

# =====================================================
# PRINT ANSWER
# =====================================================
print("\n=======================")
print("CORRECT ANSWER")
print("=======================\n")
print(response.text)