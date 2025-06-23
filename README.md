# Aventus Backend

This Flask application provides a backend service for Aventus, offering functionalities related to exercise feedback for pregnancy and prescription/food image analysis using AI models (Gemma and Gemini).

## Features

*   **Pregnancy Exercise Feedback**: Provides exercise feedback based on the week of pregnancy, number of sets, and time.
*   **Prescription Analysis (Gemma)**: Upload an image of a prescription, and the system will use OCR and the Gemma model to extract details like doctor's name, patient's name, and prescribed medicines in JSON format.
*   **Prescription Analysis (Gemini)**: Upload an image of a prescription, and the system will use the Gemini 1.5 Flash model to directly analyze the image and extract details in JSON format.
*   **Food Review (Gemini 2.0 Flash)**: Upload an image of a meal. The system uses Gemini 2.0 Flash to identify food items, then uses a RAG approach with `food_dataset.csv` to fetch nutritional information, and Gemma to provide a brief review.
*   **Food Review (Gemini 1.5 Flash)**: Similar to the above, but specifically uses Gemini 1.5 Flash for food item identification.

## Prerequisites

1.  **Python**: Python 3.8 or higher is recommended.
2.  **Tesseract OCR**: Required for the Gemma-based prescription analysis endpoint.
    *   **Installation**: Follow the official Tesseract installation guide for your operating system: [https://tesseract-ocr.github.io/tessdoc/Installation.html](https://tesseract-ocr.github.io/tessdoc/Installation.html)
    *   **Path Configuration**: Ensure Tesseract is in your system's PATH, or update the `pytesseract.pytesseract.tesseract_cmd` path in `app.py` if necessary.
        ```python
        # Example for Windows in app.py
        # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        ```
3.  **Google API Key**:
    *   Required for using Gemini models.
    *   Set up a project in Google AI Studio or Google Cloud Console.
    *   Enable the "Generative Language API" (sometimes referred to as Gemini API).
    *   Create an API key.
    *   **Important**: Ensure your API key has permissions to use the specific models:
        *   `gemini-1.5-flash` (for `/gemini_food_review` and `/parse_prescription_gemini`)
        *   The model specified by `GEMINI_MODEL_NAME` in `app.py` (default: `"gemini-2.0-flash"`, used by `/food_review`)
4.  **Hugging Face Account/Token (Optional but Recommended)**:
    *   The `transformers` library might require a Hugging Face token for downloading models like Gemma, especially if you encounter download issues. It's good practice to log in via `huggingface-cli login`.

## Setup

1.  **Clone the repository (if applicable) or ensure you have the project files.**

2.  **Create a virtual environment (recommended)**:
    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    (You will need to create a `requirements.txt` file. See section below.)

4.  **Environment Variables**:
    Create a `.env` file in the root of your project directory and add your Google API key:
    ```env
    GEMINI_API_KEY=YOUR_GOOGLE_API_KEY_HERE
    ```

5.  **Datasets**:
    *   Ensure `exercise_dataset.csv` is present in the root directory for the exercise feedback feature.
    *   Ensure `food_dataset.csv` is present in the root directory for the food review features.

## Creating `requirements.txt`

Based on the imports in `app.py`, your `requirements.txt` should look something like this. You might need to adjust versions based on your setup or if you encounter compatibility issues.

```txt
flask
pandas
numpy
faiss-cpu # or faiss-gpu if you have a compatible GPU and CUDA setup
langchain-huggingface
langchain-community
torch
transformers
python-dotenv
pytesseract
Pillow
google-generativeai
werkzeug
sentence-transformers
```
**Note**: `faiss-cpu` is generally easier to install. If you have a CUDA-enabled GPU and want to use `faiss-gpu`, ensure your CUDA toolkit and PyTorch with CUDA support are correctly installed.

## Running the Application

1.  Activate your virtual environment (if you created one).
2.  Run the Flask app:
    ```bash
    python app.py
    ```
3.  The application will start, typically on `http://0.0.0.0:8000/`.

## API Endpoints

### `/`
*   **Method**: `GET`
*   **Description**: Renders a simple HTML page (if `index.html` is present in a `templates` folder).

### `/feedback`
*   **Method**: `POST`
*   **Description**: Provides exercise feedback for pregnant women.
*   **Request Body (JSON)**:
    ```json
    {
        "week_pregnancy": "integer",
        "n_sets": "integer",
        "time": "string (e.g., 'morning', 'afternoon')"
    }
    ```
*   **Response (JSON)**:
    ```json
    {
        "feedback": "Generated exercise feedback string"
    }
    ```

### `/parse_prescription` (Gemma-based)
*   **Method**: `POST`
*   **Description**: Analyzes a prescription image using OCR and the Gemma model.
*   **Request Body (form-data)**:
    *   `image`: The prescription image file.
*   **Response (JSON)**: Expected JSON structure as defined in the Gemma prompt within `app.py` (patient info, medications, doctor info).

### `/parse_prescription_gemini` (Gemini 1.5 Flash-based)
*   **Method**: `POST`
*   **Description**: Analyzes a prescription image directly using the Gemini 1.5 Flash model.
*   **Request Body (form-data)**:
    *   `image`: The prescription image file.
*   **Response (JSON)**: Expected JSON structure as defined in the Gemini prompt within `app.py` (patient info, medications, doctor info).

### `/food_review` (Gemini 2.0 Flash for recognition)
*   **Method**: `POST`
*   **Description**: Identifies food items in an image using Gemini 2.0 Flash (or `GEMINI_MODEL_NAME`), fetches nutritional data via RAG from `food_dataset.csv`, and generates a review using Gemma.
*   **Request Body (form-data)**:
    *   `image`: The food image file.
*   **Response (JSON)**:
    ```json
    {
        "items": [
            {
                "name": "identified_food_item_1",
                "id": "1",
                "calories": "value",
                "protein": "value",
                // ... other nutritional info ...
                "matched_to": "dataset_food_name_or_estimated"
            }
            // ... other items ...
        ],
        "total_nutrition": {
            "calories": "total_calories",
            "protein": "total_protein"
        },
        "review": "Generated meal review string or empty string if review generation failed"
    }
    ```

### `/gemini_food_review` (Gemini 1.5 Flash for recognition)
*   **Method**: `POST`
*   **Description**: Identifies food items in an image specifically using Gemini 1.5 Flash, fetches nutritional data via RAG from `food_dataset.csv`, and generates a review using Gemma.
*   **Request Body (form-data)**:
    *   `image`: The food image file.
*   **Response (JSON)**: Same structure as `/food_review`.


## Notes
*   The vector stores for `exercise_dataset.csv` and `food_dataset.csv` are built in memory the first time the relevant features are accessed after the app starts. This might cause a slight delay on the first request to these endpoints.
*   Error handling is in place, and endpoints should return JSON error messages if issues occur. Check the Flask console for more detailed server-side logs.
*   The quality of OCR and AI model responses can vary based on image quality and the complexity of the input. 