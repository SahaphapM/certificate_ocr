import pytesseract
from PIL import Image
from preprocess import preprocess_image,crop_image
from pythainlp.util import normalize
import logging
import re
import requests
from fuzzywuzzy import fuzz




# ตั้งค่าพาธของ Tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Windows path


logger = logging.getLogger(__name__)

def extract_fields_from_image(image: Image.Image, studentName: str, courseName: str, cer_type: str) -> dict:
    """
    Extract relevant fields from the certificate image
    """
    # Preprocess the image
    preprocessed_image = preprocess_image(image)
    
    # Perform OCR to get the text
    full_text = pytesseract.image_to_string(preprocessed_image, lang='eng+tha')
    
    # Normalize Thai text
    full_text = normalize(full_text)

    logger.info(f"🧠 OCR Full Text:\n{full_text}")

    if cer_type == "buumooc":
        # Extract URL
        url = extract_url_from_cropped_image(preprocessed_image,cer_type)

        # Check if URL matches name and course name
        isNameMatch, isCourseMatch = url_matching(url, studentName, courseName)

    elif cer_type == "thaimooc":
        # Remove all \n in full_text
        full_text = full_text.replace("\n", " ")  # ใช้ space แทน \n เพื่อไม่ให้ข้อความติดกันมากเกินไป

        # Fuzzy Matching สำหรับการตรวจจับชื่อ
        name_match_score = fuzz.partial_ratio(studentName.lower(), full_text.lower())
        logger.info(f"🧠 Fuzzy Matching Name Score: {name_match_score}")
        isNameMatch = name_match_score >= 90  # ตั้งค่า threshold ไว้ที่ 90% สำหรับการจับคู่ชื่อ

        # Fuzzy Matching สำหรับชื่อหลักสูตร
        course_match_score = fuzz.partial_ratio(courseName.lower(), full_text.lower())
        logger.info(f"🧠 Fuzzy Matching Course Score: {course_match_score}")
        isCourseMatch = course_match_score >= 90  # ตั้งค่า threshold ไว้ที่ 90% สำหรับการจับคู่ชื่อหลักสูตร
 


    return {
            "student_name": studentName,
            "course_name": courseName,
            "cer_type": cer_type,
            "url": url,
            "isNameMatch": isNameMatch,
            "isCourseMatch": isCourseMatch,
            "full_text": full_text,
        }

def extract_url_from_cropped_image(image: Image.Image,cer_type: str) -> str:
    """
    Perform OCR on the cropped image to extract the URL
    """
    # Crop the image to focus on the bottom-left portion
    cropped_image = crop_image(image)

    # Perform OCR to get the text
    full_text = pytesseract.image_to_string(cropped_image, lang='eng+tha')

   # ดึง Certificate ID Number จาก full_text โดยดึงข้อความหลัง Certificate ID Number :
    url_match = re.search(r'Certificate ID Number\s*:\s*([^\n\r]+)', full_text)

    if url_match:
        url = url_match.group(1).strip()
        # Clean by removing inner spaces
        url = re.sub(r'\s+', '', url)
        # If it's an ID, convert to full URL for known types
        if not url.startswith('http'):
            if cer_type == "buumooc":
                url = 'https://mooc.buu.ac.th/certificates/' + url
        return url
    else:
        # Fallback: find a plain URL in the text
        fallback = re.search(r'https?://[^\s]+', full_text)
        return fallback.group(0) if fallback else None

    # Match URL to name and course name
def url_matching(url: str, studentName: str, courseName: str) -> bool:
    try:
        response = requests.get(url)
        html = response.text
        isNameMatch = studentName in html
        isCourseMatch = courseName in html

        return isNameMatch,isCourseMatch
    except Exception as e:
        logger.error(f"Error matching URL: {str(e)}")
        return False,False
