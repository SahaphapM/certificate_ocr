from PIL import Image, ImageEnhance


def preprocess_image(image: Image.Image, upscale: int = 5, contrast: float = 1.2, threshold: int = 160) -> Image.Image:
    """
    Preprocess image for better OCR results
    Args:
        image: Input PIL Image
        upscale: Image upscaling factor
        contrast: Contrast enhancement factor
        threshold: Binarization threshold
    Returns:
        Preprocessed PIL Image
    """
    image = image.convert('L')  # Convert to grayscale
    
    if upscale > 1:
        image = image.resize((image.width * upscale, image.height * upscale), Image.LANCZOS)
    
    if contrast != 1.0:
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(contrast)
    
    image = image.point(lambda x: 0 if x < threshold else 255)  # Apply thresholding
    
    image.save("img/preprocess/preprocessed_image.jpg")
    
    return image
