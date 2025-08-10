from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import io
from PIL import Image, ImageEnhance
import numpy as np
import logging
import re
import traceback
import pytesseract
from pythainlp.util import normalize
from typing import List, Tuple, Optional

# =====( Optional: PDF backends )=====
# Primary: pdf2image (Poppler required on Windows)
try:
    from pdf2image import convert_from_bytes  # type: ignore
    _HAS_PDF2IMAGE = True
except Exception:
    _HAS_PDF2IMAGE = False

# Fallback: PyMuPDF (no Poppler needed)
try:
    import fitz  # PyMuPDF
    _HAS_PYMUPDF = True
except Exception:
    _HAS_PYMUPDF = False

app = FastAPI(
    title="Thai Certificate OCR API",
    description="API for extracting information from Thai certificates using Tesseract OCR with Thai language support",
    version="1.1.0"
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

# Configure Tesseract for Thai language (adjust for your environment)
try:
    # Windows (comment out if using Linux/Mac and tesseract is in PATH)
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    logger.info("Tesseract OCR initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Tesseract OCR: {str(e)}")
    raise

# ===================== Image utils =====================

def preprocess_image(image: Image.Image, save_path: str = None, upscale: int = 2, contrast: float = 1.2, threshold: int = 160) -> Image.Image:
    """
    Preprocess image for better OCR results.
    """
    image = image.convert('L')  # grayscale

    # Upscale
    if upscale and upscale > 1:
        image = image.resize((image.width * upscale, image.height * upscale), Image.LANCZOS)

    # Enhance contrast
    if contrast and contrast != 1.0:
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(contrast)

    # Binarize
    image = image.point(lambda x: 0 if x < threshold else 255)

    if save_path:
        try:
            image.save(save_path)
        except Exception as e:
            logger.warning(f"Could not save preprocessed image {save_path}: {e}")

    return image

def extract_url_from_image(image: Image.Image) -> str:
    """
    Extract URL from the certificate (tries bottom strip first, then whole image).
    """
    width, height = image.size
    crop_box = (0, int(height * 0.92), width, height)  # 8% ท้ายภาพ จะครอบกว้างขึ้นเล็กน้อย
    cropped = image.crop(crop_box)

    cropped = preprocess_image(
        cropped,
        save_path=None,
        upscale=2,
        contrast=1.0,
        threshold=150
    )

    url_text = pytesseract.image_to_string(cropped, lang='tha+eng')
    url_text_clean = re.sub(r'[\s|]', '', url_text or '')

    # Use regex to extract a URL without page numbers or extra info
    match = re.search(r'https?://[a-zA-Z0-9.-]+(?:/[a-zA-Z0-9.-]+)*', url_text_clean)

    if match:
        url = url_text_clean[url_text_clean.find('http'):]

        # Clean the URL to remove unwanted page number info (e.g., 1/2)
        url = re.sub(r'\s*\d+/\d+\s*$', '', url)  # Remove page numbers like '1/2' at the end

        url = url.replace(':/', '://').replace(' ', '').replace('|', '')
        url = re.split(r'[^a-zA-Z0-9:/._\-]', url)[0]
        return url

    # Fallback: process the whole image if URL wasn't found in the cropped part
    full_img = preprocess_image(
        image,
        save_path=None,
        upscale=2,
        contrast=1.2,
        threshold=160
    )
    full_text = pytesseract.image_to_string(full_img, lang='tha+eng')
    full_text_clean = re.sub(r'[\s|]', '', full_text or '')

    match_full = re.search(r'https?://[a-zA-Z0-9.-]+(?:/[a-zA-Z0-9.-]+)*', full_text_clean)
    if match_full:
        url = full_text_clean[full_text_clean.find('http'):]

        # Clean the URL to remove unwanted page number info (e.g., 1/2)
        url = re.sub(r'\s*\d+/\d+\s*$', '', url)  # Remove page numbers like '1/2' at the end

        url = url.replace(':/', '://').replace(' ', '').replace('|', '')
        url = re.split(r'[^a-zA-Z0-9:/._\-]', url)[0]
        return url

    return ''

def extract_fields_from_text(full_text: str) -> Tuple[str, str, str]:
    """
    Extract student_name, course_name, date from OCR text by regex.
    """
    # log original text
    logger.info("🧠 Original Text:\n" + (full_text or ""))
    full_text = normalize(full_text or "")
    logger.info("🧠 OCR Full Text (normalized):\n" + full_text)

    # Name
    name_match = (
        re.search(r"มอบให้\s+(.+)", full_text) or
        re.search(r"presented to\s+(.+)", full_text, re.IGNORECASE)
    )
    student_name = name_match.group(1).strip() if name_match else ""

    # Course
    course_match = (
        re.search(r"หลักสูตร\s+(.+)", full_text) or
        re.search(r"completed the Open Online Course\s+(.+)", full_text, re.IGNORECASE)
    )
    course_name = course_match.group(1).strip() if course_match else ""

    # Date
    date_match = (
        re.search(r"วันที่\s+(\d{1,2}\s+[ก-๙]+\s+\d{4})", full_text) or
        re.search(r"On\s+([A-Za-z]+\s+\d{1,2},\s+\d{4})", full_text, re.IGNORECASE)
    )
    course_date = date_match.group(1).strip() if date_match else ""

    return student_name, course_name, course_date

def ocr_on_image(image: Image.Image) -> Tuple[str, str]:
    """
    Run OCR on the whole image after preprocessing, returns (full_text, url).
    """
    preprocessed_image = preprocess_image(
        image,
        save_path=None,
        upscale=2,
        contrast=1.2,
        threshold=160
    )
    image_np = np.array(preprocessed_image)
    full_text = pytesseract.image_to_string(image_np, lang='tha+eng') or ""
    url = extract_url_from_image(image)
    return full_text, url

# ===================== PDF to Images =====================

def _pdf_to_images_pdf2image(contents: bytes, dpi: int = 300) -> List[Image.Image]:
    """
    Convert PDF bytes to list of PIL Images using pdf2image.
    Requires Poppler (on Windows set poppler_path to your poppler bin folder if needed).
    """
    # You may set poppler_path explicitly on Windows:
    # pages = convert_from_bytes(contents, dpi=dpi, poppler_path=r"C:\path\to\poppler\Library\bin")
    pages = convert_from_bytes(contents, dpi=dpi)
    return pages

def _pdf_to_images_pymupdf(contents: bytes, zoom: float = 2.0) -> List[Image.Image]:
    """
    Convert PDF bytes to list of PIL Images using PyMuPDF (fitz). No Poppler needed.
    """
    images: List[Image.Image] = []
    with fitz.open(stream=contents, filetype="pdf") as doc:
        mat = fitz.Matrix(zoom, zoom)  # 2.0 ~ ~144-200 dpi-ish; adjust as needed
        for page in doc:
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img)
    return images

def pdf_to_images(contents: bytes) -> List[Image.Image]:
    """
    Try pdf2image first, fall back to PyMuPDF.
    """
    if _HAS_PDF2IMAGE:
        try:
            return _pdf_to_images_pdf2image(contents, dpi=300)
        except Exception as e:
            logger.warning(f"pdf2image failed: {e}. Falling back to PyMuPDF...")
    if _HAS_PYMUPDF:
        try:
            return _pdf_to_images_pymupdf(contents, zoom=2.0)
        except Exception as e:
            logger.error(f"PyMuPDF also failed: {e}")
            raise
    raise HTTPException(
        status_code=500,
        detail="No PDF renderer available. Install either 'pdf2image' (with Poppler) or 'PyMuPDF'."
    )

def is_pdf(content_type: Optional[str], filename: Optional[str]) -> bool:
    """
    Robust PDF check (content_type can be unreliable from some clients).
    """
    if content_type and content_type.lower() in {"application/pdf", "application/x-pdf"}:
        return True
    if filename and filename.lower().endswith(".pdf"):
        return True
    return False

# ===================== Extract & Merge Helpers =====================

def merge_fields(pages_data: List[dict]) -> dict:
    """
    Merge multi-page fields: take the first non-empty of each, concatenate full_text, pick first URL found.
    """
    student_name = ""
    course_name = ""
    date = ""
    url = ""
    texts = []

    for d in pages_data:
        texts.append(d.get("full_text", "") or "")
        if not student_name and d.get("student_name"):
            student_name = d["student_name"]
        if not course_name and d.get("course_name"):
            course_name = d["course_name"]
        if not date and d.get("date"):
            date = d["date"]
        if not url and d.get("url"):
            url = d["url"]

    return {
        "student_name": student_name,
        "course_name": course_name,
        "date": date,
        "url": url,
        "full_text": "\n\n".join(t for t in texts if t)
    }

# ===================== FastAPI Endpoint =====================

@app.post("/ocr", summary="Extract information from Thai certificate image/PDF")
async def ocr_certificate(file: UploadFile = File(...)):
    """
    Process an uploaded image or PDF and extract relevant certificate information.
    """
    logger.info(f"🚀 Processing OCR request for file: {file.filename} (type={file.content_type})")

    try:
        # Read all bytes ONCE
        contents: bytes = await file.read()
        size = len(contents)
        logger.info(f"📥 Received file: {file.filename} ({size} bytes)")

        # Validate size (10 MB)
        if size > 10 * 1024 * 1024:
            raise HTTPException(
                status_code=400,
                detail="File size exceeds the limit of 10MB"
            )

        # PDF flow
        if is_pdf(file.content_type, file.filename):
            if not contents:
                raise HTTPException(status_code=400, detail="Empty PDF file.")
            try:
                pages: List[Image.Image] = pdf_to_images(contents)
                if not pages:
                    raise HTTPException(status_code=400, detail="No pages found in PDF.")
                logger.info(f"PDF converted to {len(pages)} page image(s).")

                per_page_results = []
                for idx, page_img in enumerate(pages, start=1):
                    try:
                        full_text, url = ocr_on_image(page_img)
                        name, course, dt = extract_fields_from_text(full_text)
                        per_page_results.append({
                            "page": idx,
                            "student_name": name,
                            "course_name": course,
                            "date": dt,
                            "url": url,
                            "full_text": full_text
                        })
                    except Exception as e:
                        logger.warning(f"OCR failed on page {idx}: {e}")
                        per_page_results.append({
                            "page": idx,
                            "student_name": "",
                            "course_name": "",
                            "date": "",
                            "url": "",
                            "full_text": ""
                        })

                merged = merge_fields(per_page_results)
                logger.info("✅ Fields extracted from PDF")
                return {
                    "status": "success",
                    "source_type": "pdf",
                    "pages": len(pages),
                    "data": merged,
                    "per_page": per_page_results  # useful for debugging/QA
                }
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Failed to process PDF: {e}")
                logger.error(traceback.format_exc())
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to process PDF: {str(e)}"
                )

        # Image flow
        else:
            try:
                image = Image.open(io.BytesIO(contents))
                logger.info(f"Image loaded successfully: {image.size} {image.mode}")
            except Exception as e:
                logger.error(f"Failed to load image: {str(e)}")
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to load image: {str(e)}"
                )

            try:
                full_text, url = ocr_on_image(image)
                name, course, dt = extract_fields_from_text(full_text)
                fields = {
                    "student_name": name,
                    "course_name": course,
                    "date": dt,
                    "url": url,
                    "full_text": full_text
                }
                logger.info("✅ Fields extracted successfully (image)")
                return {
                    "status": "success",
                    "source_type": "image",
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
    uvicorn.run("main_tesseract_upgrade:app", host="127.0.0.1", port=8000, reload=True)
