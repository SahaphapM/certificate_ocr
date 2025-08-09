How to run

1. Create virtual environment with python -m venv .venv
2. Activate virtual environment with .venv\Scripts\activate
3. Install dependencies with pip install -r requirements.txt
4. Run the app with uvicorn main:app --reload

How to use

1. Send POST request to /ocr
2. Include studentName, courseName, cer_type in the request body
3. Include file in the request body

