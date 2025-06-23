from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import io
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import logging
import re
import traceback
import pytesseract
from pythainlp.util import normalize

app = FastAPI(
    title="Thai Certificate OCR API",
    description="API for extracting information from Thai certificates using Tesseract OCR with Thai language support",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configure Tesseract for Thai language
try:
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Windows path
    logger.info("Tesseract OCR initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Tesseract OCR: {str(e)}")
    raise

def preprocess_image(image: Image.Image, save_path: str = None, upscale: int = 2, contrast: float = 1.2, threshold: int = 160) -> Image.Image:
    """
    Preprocess image for better OCR results
    Args:
        image: Input PIL Image
        save_path: Optional path to save preprocessed image
        upscale: Image upscaling factor
        contrast: Contrast enhancement factor
        threshold: Binarization threshold
    Returns:
        Preprocessed PIL Image
    """
    # Convert to grayscale
    image = image.convert('L')
    
    # Upscale image
    if upscale > 1:
        image = image.resize((image.width * upscale, image.height * upscale), Image.LANCZOS)
    
    # Enhance contrast
    if contrast != 1.0:
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(contrast)
    
    # Apply threshold
    image = image.point(lambda x: 0 if x < threshold else 255)
    
    # Save if path provided
    if save_path:
        image.save(save_path)
    
    return image

def extract_url_from_image(image: Image.Image) -> str:
    """
    Extract URL from the bottom portion of the certificate
    """
    width, height = image.size
    # Crop bottom 5% of the image where URL is typically located
    crop_box = (0, int(height * 0.95), width, height)
    cropped = image.crop(crop_box)
    
    # Preprocess the cropped area
    cropped = preprocess_image(
        cropped,
        save_path='debug_url_zone_preprocessed.png',
        upscale=2,
        contrast=1.0,
        threshold=150
    )
    
    # Perform OCR on the cropped area
    url_text = pytesseract.image_to_string(cropped, lang='tha+eng')
    url_text_clean = url_text.replace(' ', '').replace('|', '').replace('\n', '').replace('\r', '')
    
    # Try to find URL in the cropped area
    match = re.search(r'https?[:/ ]+([a-zA-Z0-9.\-_]+)[/a-zA-Z0-9.\-_]+', url_text_clean)
    if match:
        url = url_text_clean[url_text_clean.find('http'):]
        url = url.replace(':/', '://').replace(' ', '').replace('|', '')
        url = re.split(r'[^a-zA-Z0-9:/._\-]', url)[0]
        return url
    
    # If URL not found in cropped area, try the whole image
    full_img = preprocess_image(
        image,
        save_path='debug_full_preprocessed.png',
        upscale=2,
        contrast=1.2,
        threshold=160
    )
    full_text = pytesseract.image_to_string(full_img, lang='tha+eng')
    full_text_clean = full_text.replace(' ', '').replace('|', '').replace('\n', '').replace('\r', '')
    
    match_full = re.search(r'https?[:/ ]+([a-zA-Z0-9.\-_]+)[/a-zA-Z0-9.\-_]+', full_text_clean)
    if match_full:
        url = full_text_clean[full_text_clean.find('http'):]
        url = url.replace(':/', '://').replace(' ', '').replace('|', '')
        url = re.split(r'[^a-zA-Z0-9:/._\-]', url)[0]
        return url
    
    return ''

def extract_fields_from_image(image: Image.Image) -> dict:
    """
    Extract all relevant fields from the certificate image
    """
    # Preprocess the entire image
    preprocessed_image = preprocess_image(
        image,
        save_path='debug_full_preprocessed.png',
        upscale=2,
        contrast=1.2,
        threshold=160
    )
    
    # Perform OCR on the preprocessed image
    image_np = np.array(preprocessed_image)
    full_text = pytesseract.image_to_string(image_np, lang='tha+eng')
    # log original text
    logger.info("🧠 Original Text:\n" + full_text)
    full_text = normalize(full_text)  # Normalize Thai text
    logger.info("🧠 OCR Full Text:\n" + full_text)

    # Extract fields using regex patterns
    # Thai patterns
    name_match = re.search(r"มอบให้\s+(.+)", full_text) or re.search(r"presented to\s+(.+)", full_text, re.IGNORECASE)
    student_name = name_match.group(1).strip() if name_match else ""

    course_match = re.search(r"หลักสูตร\s+(.+)", full_text) or re.search(r"completed the Open Online Course\s+(.+)", full_text, re.IGNORECASE)
    course_name = course_match.group(1).strip() if course_match else ""

    date_match = re.search(r"วันที่\s+(\d{1,2}\s+[ก-๙]+\s+\d{4})", full_text) or re.search(r"On\s+([A-Za-z]+\s+\d{1,2},\s+\d{4})", full_text, re.IGNORECASE)
    course_date = date_match.group(1).strip() if date_match else ""

    # Extract URL
    url = extract_url_from_image(image)

    fields = {
        "student_name": student_name,
        "course_name": course_name,
        "date": course_date,
        "url": url,
        "full_text": full_text
    }
    
    logger.info("✅ Fields Extracted: " + str(fields))
    return fields

@app.post("/ocr", summary="Extract information from Thai certificate image")
async def ocr_certificate(file: UploadFile = File(...)):
    """
    Process a certificate image and extract relevant information
    """
    logger.info(f"🚀 Processing OCR request for file: {file.filename}")
    
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type: {file.content_type}. Only image files are allowed."
            )
        
        # Read file contents
        contents = await file.read()
        logger.info(f"📥 Received file: {file.filename} ({len(contents)} bytes)")
        
        try:
            # Load image
            image = Image.open(io.BytesIO(contents))
            logger.info(f"Image loaded successfully: {image.size} {image.mode}")
        except Exception as e:
            logger.error(f"Failed to load image: {str(e)}")
            raise HTTPException(
                status_code=400,
                detail=f"Failed to load image: {str(e)}"
            )
        
        try:
            # Extract fields
            fields = extract_fields_from_image(image)
            logger.info("✅ Fields extracted successfully")
            return {
                "status": "success",
                "data": fields
            }
        except Exception as e:
            logger.error(f"Failed to extract fields: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(
                status_code=500,
                detail=f"Failed to extract fields from image: {str(e)}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main_tesseract_upgrade:app", host="0.0.0.0", port=8000, reload=True) 