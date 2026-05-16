# =====================================================
# IMPORTS
# =====================================================
import cv2
import pytesseract
from PIL import Image
import os
from google import genai

print("Installed Successfully")

# =====================================================
# GEMINI API KEY (Render safe)
# =====================================================
api_key = os.environ.get("GOOGLE_API_KEY")

if not api_key:
    print("ERROR: GOOGLE_API_KEY not set")
    exit()

client = genai.Client(api_key=api_key)

# =====================================================
# TESSERACT PATH (Render-safe handling)
# =====================================================
if os.name == "nt":  # Windows
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# =====================================================
# IMAGE PATH (GitHub / Render safe)
# =====================================================
image_path = "21.jpg"   # or "images/21.jpg"

img = cv2.imread(image_path)

if img is None:
    print("ERROR: Image not found!")
    exit()

# =====================================================
# IMAGE PREPROCESSING
# =====================================================
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)[1]

processed_path = "processed.png"
cv2.imwrite(processed_path, thresh)

# =====================================================
# OCR
# =====================================================
text = pytesseract.image_to_string(Image.open(processed_path))

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
# GEMINI RESPONSE (NEW SDK)
# =====================================================
response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=prompt
)

# =====================================================
# OUTPUT
# =====================================================
print("\n=======================")
print("CORRECT ANSWER")
print("=======================\n")

print(response.text)