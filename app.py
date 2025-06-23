import pandas as pd
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS  # type: ignore
from transformers import pipeline
from flask import Flask, request, jsonify, render_template
import torch
import os  # For file operations
from werkzeug.utils import secure_filename  # For secure filenames
import pytesseract  # For OCR
# Moved ImageEnhance, ImageFilter to global imports
from PIL import Image, ImageEnhance, ImageFilter
import google.generativeai as genai  # Added for Gemini
import io  # Added import
import json  # Added import

from dotenv import load_dotenv

load_dotenv()

# --- Configuration for Tesseract (Optional, if not in PATH) ---
# If Tesseract is not in your system PATH, uncomment and set the path below:
# Example for Windows
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# --- Configuration for Gemini ---
# Ensure your GOOGLE_API_KEY environment variable is set.
# genai.configure(api_key=os.environ["GOOGLE_API_KEY"]) # This will be called before first use.
# Using a common model name, adjust if you have a specific one like "gemini-1.5-flash-001"
GEMINI_MODEL_NAME = "gemini-2.0-flash"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")  # User provided API Key

# --- Upload Folder Configuration ---
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tiff', 'bmp', 'webp'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize Gemini client (do it once)
try:
    genai.configure(api_key=GEMINI_API_KEY)  # Use the hardcoded API key
    gemini_model_client = genai.GenerativeModel(GEMINI_MODEL_NAME)
    print("Gemini client configured successfully with the provided API key.")
except Exception as e:  # General exception for any configuration errors
    gemini_model_client = None
    print(
        f"Error configuring Gemini client with the provided API key: {e}. Gemini functionality will be disabled.")


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Load CSV
df = pd.read_csv("exercise_dataset.csv")

# Create descriptions from the dataset fields for vector embeddings
# Make descriptions more specific with structured format
descriptions = []
for _, row in df.iterrows():
    desc = f"Exercise name: {row['name']}, Pregnancy week: {row['week']}, Time of day: {row['time of day']}, " \
           f"Sets/reps: {row['N']}, Benefits: {row['benefits']}"
    descriptions.append(desc)

# Initialize embeddings and FAISS
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2")
texts = descriptions
metadata_list = [
    {
        "name": row['name'],
        "week": row['week'],
        "time": row['time of day'],
        "N": row['N'],
        "benefits": row['benefits'],
        "link": row['link'],
        "row_index": i
    } for i, row in df.iterrows()
]
vectorstore = FAISS.from_texts(texts, embeddings, metadatas=metadata_list)

# Initialize Gemma
gemma = pipeline(
    "text-generation",
    model="google/gemma-3-4b-it",
    device=0 if torch.cuda.is_available() else -1,
    torch_dtype=torch.bfloat16
)

# Renamed for clarity, can be used by both Gemma and Gemini if they follow the format


def parse_llm_prescription_output(text):
    output = {
        "patient_name": None,  # Swapped order to match your new request
        "doctor_name": None,
        "age": None,  # Age and Sex are still good to have if present
        "sex": None,
        "prescribed_medicines": []  # Renamed for clarity
    }
    lines = text.split('\n')
    current_medicines_section = False
    # Keywords to look for, making it a bit more flexible
    key_map = {
        "patient's name": "patient_name",
        "doctor's name": "doctor_name",
        "age": "age",
        "sex": "sex"
    }

    for line in lines:
        line_lower = line.lower().strip()
        if not line_lower:
            continue

        processed_field = False
        for key, json_key in key_map.items():
            if line_lower.startswith(key + ":"):
                output[json_key] = line.split(":", 1)[1].strip()
                processed_field = True
                break
        if processed_field:
            continue

        if line_lower.startswith("medicines:"):
            current_medicines_section = True
        elif current_medicines_section and (line_lower.startswith("- name:") or line_lower.startswith("name:")):
            med_info = {"name": None, "dosage": None,
                        "disease_inference": None}  # Added disease_inference
            # Example: "- Name: Aspirin, Dosage: 1 tablet daily, Disease Inference: Pain, Fever, Inflammation"
            content_line = line.split(
                ":", 1)[1].strip() if ":" in line else line.strip()

            parts = [p.strip() for p in content_line.split(',')]

            current_med_name = None
            current_med_dosage = None
            current_disease_inference = None

            for part in parts:
                part_lower = part.lower()
                if part_lower.startswith("dosage:"):
                    current_med_dosage = part.split(":", 1)[1].strip()
                elif part_lower.startswith("disease inference:"):
                    current_disease_inference = part.split(":", 1)[1].strip()
                # Name should be the first part if not explicitly prefixed, or if other known prefixes are not met
                # Avoid capturing old 'use:' as name
                elif current_med_name is None and not part_lower.startswith("use:"):
                    current_med_name = part.strip()

            if current_med_name:  # Ensure name is found before adding
                med_info["name"] = current_med_name
                med_info["dosage"] = current_med_dosage
                med_info["disease_inference"] = current_disease_inference
                output["prescribed_medicines"].append(
                    med_info)  # Changed key name

        elif current_medicines_section and not (line_lower.startswith("- ") or line_lower.startswith("name:")):
            current_medicines_section = False

    return output


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/feedback", methods=["POST"])
def get_feedback():
    data = request.get_json()
    week_pregnancy = data.get("week_pregnancy")
    n_sets = data.get("n_sets")
    time = data.get("time")
    name = data.get("name", "")

    if not all([week_pregnancy, n_sets, time]):
        return jsonify({"error": "Missing parameters"}), 400

    try:
        week_pregnancy = int(week_pregnancy)
        n_sets = int(n_sets)
    except ValueError:
        return jsonify({"error": "week_pregnancy and n_sets must be integers"}), 400

    # First try direct filtering if name is provided (exact matching)
    exact_matches = []
    if name:
        name_lower = name.lower().strip()
        time_lower = time.lower().strip()

        # Filter dataframe for exact matches
        matches = df[(df['name'].str.lower() == name_lower)
                     & (df['week'] == week_pregnancy)]

        if not matches.empty:
            # If we have time match, prioritize it
            time_matches = matches[matches['time of day'].str.lower(
            ) == time_lower]
            filtered_matches = time_matches if not time_matches.empty else matches

            # Convert to our metadata format
            for _, row in filtered_matches.iterrows():
                exact_matches.append({
                    "name": row['name'],
                    "week": row['week'],
                    "time": row['time of day'],
                    "N": row['N'],
                    "benefits": row['benefits'],
                    "link": row['link'],
                })

    # If we found exact matches, use them exclusively
    if exact_matches:
        relevant_exercises = exact_matches
    else:
        # If no exact matches or no name provided, use RAG with enhanced query
        if name:
            query = f"Exercise name: {name}, Pregnancy week: {week_pregnancy}, Time of day: {time}"
        else:
            query = f"Exercises for pregnancy week: {week_pregnancy}, Time of day: {time}"

        # Retrieve relevant exercises using RAG
        docs = vectorstore.similarity_search(query, k=4)

        # Extract exercises from metadata
        relevant_exercises = []
        for doc in docs:
            metadata = doc.metadata

            # Apply post-retrieval filtering for better match quality
            # Prioritize exact week matches
            if 'week' in metadata and metadata['week'] == week_pregnancy:
                # Extract exercise info from metadata
                exercise_info = {
                    "name": metadata['name'],
                    "week": metadata['week'],
                    "time": metadata['time'],
                    "N": metadata['N'],
                    "benefits": metadata['benefits'],
                    "link": metadata['link']
                }
                relevant_exercises.append(exercise_info)

        # If we didn't find any exercises after filtering by week, use the top results regardless
        if not relevant_exercises and docs:
            for doc in docs:
                metadata = doc.metadata
                exercise_info = {
                    "name": metadata['name'],
                    "week": metadata['week'],
                    "time": metadata['time'],
                    "N": metadata['N'],
                    "benefits": metadata['benefits'],
                    "link": metadata['link']
                }
                relevant_exercises.append(exercise_info)

    # Format context from retrieved exercises
    context_parts = []
    for ex in relevant_exercises:
        context_parts.append(
            f"Exercise: {ex['name']} for week {ex['week']} of pregnancy\n"
            f"  - Recommended time: {ex['time']}\n"
            f"  - Recommended sets/reps: {ex['N']}\n"
            f"  - Benefits: {ex['benefits']}\n"
            f"  - Link: {ex['link']}"
        )

    context = "\n\n".join(context_parts)

    # Generate feedback with Gemma using a more structured, objective prompt
    prompt = f"""EXERCISE FEEDBACK SYSTEM

USER INFORMATION:
- Exercise: {name if name else 'Not specified'}
- Pregnancy week: {week_pregnancy}
- Sets performed: {n_sets}
- Time: {time}

REFERENCE DATA:
{context}

TASK:
Generate concise, objective exercise feedback for this pregnancy exercise routine.
The feedback should:
1. State what exercise was performed and at what pregnancy week
2. Compare user's sets ({n_sets}) with recommended sets
3. Explain exercise benefits during pregnancy
4. Provide clear safety guidelines and modifications if needed
5. Include links to resources for proper form

FORMAT REQUIREMENTS:
- Use factual, objective language (avoid "I think", "my opinion", etc.)
- Keep complete sentences
- Be concise but comprehensive
- Ensure response is complete with no cut-off sentences
"""

    # Increase max tokens to ensure we get complete responses
    generated_output = gemma(prompt, max_new_tokens=800,
                             truncation=True, return_full_text=False)

    response_content = ""
    if generated_output and isinstance(generated_output, list) and generated_output[0].get("generated_text"):
        response_content = generated_output[0]["generated_text"].strip()

    return jsonify({"feedback": response_content})

# Common function for OCR and file handling


def process_uploaded_image():
    if not request.files:  # Check if any files are part of the request
        return None, jsonify({"error": "No file part in the request. Ensure you are sending a file with multipart/form-data."}), 400

    file_to_process = None

    if 'image' in request.files:
        file_to_process = request.files['image']
        if not file_to_process.filename:  # Check if an empty file input was submitted under 'image'
            return None, jsonify({"error": "File field 'image' is present but no file was selected."}), 400
    elif len(request.files) == 1:
        # If 'image' field is not present, but there's exactly one file in the request, use that.
        file_to_process = next(iter(request.files.values()))
        if not file_to_process.filename:
            return None, jsonify({"error": "A file field is present but no file was selected (filename is empty)."}), 400
    else:
        # No 'image' field and either 0 or >1 files.
        return None, jsonify({"error": f"Ambiguous file upload. Please use the 'image' field name, or send only one file if not using 'image' field. Found fields: {list(request.files.keys())}"}), 400

    if not allowed_file(file_to_process.filename):
        return None, jsonify({"error": f"File type not allowed. Allowed types are: {', '.join(ALLOWED_EXTENSIONS)}. Got: {file_to_process.filename}"}), 400

    filename = secure_filename(file_to_process.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    try:
        file_to_process.save(filepath)

        # --- Image Enhancement ---
        img = Image.open(filepath)

        # 1. Convert to Grayscale
        img = img.convert('L')

        # 2. Enhance Contrast (Pillow's ImageEnhance module)
        enhancer_contrast = ImageEnhance.Contrast(img)
        # Factor 1.0 is original, >1 increases contrast
        img = enhancer_contrast.enhance(1.5)

        # 3. Sharpen (Pillow's ImageFilter module)
        img = img.filter(ImageFilter.SHARPEN)

        # --- OCR ---
        # extracted_text = pytesseract.image_to_string(Image.open(filepath)) # Old way
        extracted_text = pytesseract.image_to_string(
            img)  # New way with enhanced image

        if not extracted_text.strip():
            # Try OCR with original image as a fallback if enhanced image yields nothing
            try:
                original_img_for_fallback_ocr = Image.open(filepath)
                fallback_text = pytesseract.image_to_string(
                    original_img_for_fallback_ocr)
                if fallback_text.strip():
                    extracted_text = fallback_text
                    print(
                        "OCR with enhanced image failed, but succeeded with original image.")
                else:
                    return None, jsonify({"error": "OCR could not extract any text from the image (enhanced or original)."}), 400
            except Exception as ocr_fallback_err:
                print(f"Error during fallback OCR attempt: {ocr_fallback_err}")
                return None, jsonify({"error": "OCR could not extract any text from the image, and fallback OCR also failed."}), 400

        return extracted_text, None, None  # Success
    except pytesseract.TesseractNotFoundError:
        # The Tesseract path is configured at the top of the file.
        # If this error occurs, it means that path is incorrect or Tesseract is not installed correctly.
        return None, jsonify({"error": "Tesseract is not installed correctly or tesseract_cmd path in app.py is wrong."}), 500
    except Exception as e:
        # Log the exception for server-side debugging
        print(f"Error during image processing or OCR: {e}")
        import traceback
        traceback.print_exc()
        return None, jsonify({"error": "Error during image processing or OCR", "details": str(e)}), 500
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)


@app.route("/parse_prescription", methods=["POST"])
def parse_prescription_gemma_endpoint():
    # --- Identical Image Input Handling (as in Gemini endpoint) ---
    image_file = None
    if 'image' in request.files:
        image_file = request.files['image']
    elif len(request.files) == 1:
        image_file = next(iter(request.files.values()))

    if not image_file or not image_file.filename:
        return jsonify({"error": "No image file provided or filename is empty."}), 400

    if not allowed_file(image_file.filename):
        return jsonify({"error": f"File type not allowed. Allowed types are: {', '.join(ALLOWED_EXTENSIONS)}. Got: {image_file.filename}"}), 400

    try:
        pil_image_original = Image.open(image_file.stream)
        # Ensure image is in a format pytesseract can handle well (e.g. RGB, L)
        # and make a copy for enhancement.
        if pil_image_original.mode == 'P':  # Convert palettized images
            pil_image_to_enhance = pil_image_original.convert('RGB')
        else:
            pil_image_to_enhance = pil_image_original.copy()

    except Exception as e:
        print(f"Error opening or preparing image: {str(e)}")
        return jsonify({"error": f"Error opening or preparing image: {str(e)}"}), 500

    # --- Image Enhancement (applied directly to PIL image) ---
    try:
        # 1. Convert to Grayscale
        img_enhanced = pil_image_to_enhance.convert('L')
        # 2. Enhance Contrast
        enhancer_contrast = ImageEnhance.Contrast(img_enhanced)
        img_enhanced = enhancer_contrast.enhance(1.5)
        # 3. Sharpen
        img_enhanced = img_enhanced.filter(ImageFilter.SHARPEN)
    except Exception as e:
        print(f"Error during image enhancement: {str(e)}")
        return jsonify({"error": f"Error during image enhancement: {str(e)}"}), 500

    # --- OCR (applied directly to PIL image) ---
    extracted_text = ""
    try:
        extracted_text = pytesseract.image_to_string(img_enhanced)
        if not extracted_text.strip():
            print("OCR with enhanced image failed, attempting with original image.")
            # Fallback to original image (ensure it's in a suitable mode like RGB or L)
            pil_fallback = pil_image_original.copy()  # Use a fresh copy of original
            if pil_fallback.mode == 'P':
                pil_fallback = pil_fallback.convert('RGB')
            elif pil_fallback.mode not in ['L', 'RGB']:
                # Default to RGB if unknown strange mode
                pil_fallback = pil_fallback.convert('RGB')

            extracted_text = pytesseract.image_to_string(pil_fallback)
            if extracted_text.strip():
                print("Fallback OCR on original image succeeded.")
            else:
                print("Fallback OCR on original image also failed.")
                return jsonify({"error": "OCR could not extract any text from the image (enhanced or original)."}), 400

        if not extracted_text.strip():  # Final check if text is still empty
            return jsonify({"error": "OCR failed to extract text from image after all attempts."}), 400

    except pytesseract.TesseractNotFoundError:
        # This error implies Tesseract is not installed or tesseract_cmd is not set correctly.
        # The tesseract_cmd is set at the top of app.py.
        print("TesseractNotFoundError: Ensure Tesseract is installed and configured correctly.")
        return jsonify({"error": "Tesseract is not installed correctly or tesseract_cmd path in app.py is wrong."}), 500
    except Exception as e:
        print(f"Error during OCR processing: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": "Error during OCR processing", "details": str(e)}), 500

    # --- Gemma Model Processing (using OCR text, prompt asks for JSON) ---
    # Prompt is already updated from previous steps to ask for the correct JSON structure.
    prompt = f"""You are an expert medical prescription analyzer. Analyze this prescription image and extract the following information:
1. All medications listed in the prescription
2. The dosage for each medication
3. The medical condition or purpose of each medication
4. Any special instructions

Format your response as a structured JSON with the following format:
{{
  "patient_info": {{
    "name": "", 
    "date": ""
  }},
  "medications": [
    {{
      "medicine": "Medication Name 1",
      "dosage": "Dosage information",
      "purpose": "Used for treating condition",
      "instructions": "Special instructions if any"
    }}
    // ... potentially more medication objects if found
  ],
  "additional_instructions": "",
  "doctor_info": {{
    "name": "",
    "contact": ""
  }}
}}

Be sure to extract accurate information and maintain the precise JSON format. If you can\'t determine certain information, use empty strings or "Unknown" as values.

The following is the text extracted via OCR from the prescription image:
--- START OF OCR-EXTRACTED PRESCRIPTION TEXT ---
{extracted_text}
--- END OF OCR-EXTRACTED PRESCRIPTION TEXT ---

Now, provide your analysis based on this text in the JSON format specified above. Ensure the output is ONLY the JSON object, with no other text before or after it.
"""

    gemma_response_text = ""
    try:
        if gemma is None:
            print("Gemma pipeline is not initialized.")
            return jsonify({"error": "Gemma model pipeline not available."}), 503

        generated_output = gemma(
            prompt, max_new_tokens=1024, truncation=True, return_full_text=False)

        if generated_output and isinstance(generated_output, list) and generated_output[0].get("generated_text"):
            gemma_response_text = generated_output[0]["generated_text"].strip()
        else:
            print(
                f"Unexpected output structure from Gemma: {generated_output}")
            return jsonify({"error": "Gemma model returned an unexpected or empty output structure.", "raw_gemma_output": str(generated_output)}), 500

    except Exception as e:
        print(f"Error processing request with Gemma model: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": "Error processing request with Gemma model", "details": str(e)}), 500

    if not gemma_response_text:
        return jsonify({"error": "Gemma model returned no text after processing OCR content.", "ocr_extracted_text": extracted_text}), 500

    # --- JSON Parsing (already in place) ---
    try:
        json_start = gemma_response_text.find('{')
        json_end = gemma_response_text.rfind('}') + 1
        if json_start != -1 and json_end > json_start:
            json_str = gemma_response_text[json_start:json_end]
            parsed_data = json.loads(json_str)
        else:
            parsed_data = json.loads(gemma_response_text)

    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON response from Gemma: {e}")
        print(f"OCR extracted text sent to Gemma: {extracted_text}")
        print(f"Raw Gemma response: {gemma_response_text}")
        return jsonify({
            "error": "Failed to parse JSON response from Gemma model.",
            "details": str(e),
            "ocr_extracted_text": extracted_text,
            "raw_model_output": gemma_response_text
        }), 500
    except Exception as e:
        print(
            f"An unexpected error occurred while parsing Gemma response: {e}")
        print(f"Raw Gemma response: {gemma_response_text}")
        return jsonify({
            "error": "An unexpected error occurred while parsing the Gemma response.",
            "details": str(e),
            "raw_model_output": gemma_response_text
        }), 500

    if not isinstance(parsed_data, dict) or "medications" not in parsed_data:
        print(
            f"Parsed data from Gemma does not have expected structure. Data: {parsed_data}")
        return jsonify({
            "warning": "Gemma: Parsed data does not have the expected structure (e.g., missing 'medications' key).",
            "ocr_extracted_text": extracted_text,
            "raw_model_output": gemma_response_text,
            "parsed_data": parsed_data
        }), 200

    return jsonify(parsed_data)


@app.route("/parse_prescription_gemini", methods=["POST"])
def parse_prescription_gemini_endpoint():
    if not gemini_model_client:
        return jsonify({"error": "Gemini client is not configured. Check GOOGLE_API_KEY."}), 503

    image_file = None
    if 'image' in request.files:
        image_file = request.files['image']
    elif len(request.files) == 1:  # Allow if only one file is sent without 'image' field
        image_file = next(iter(request.files.values()))

    if not image_file or not image_file.filename:
        return jsonify({"error": "No image file provided or filename is empty."}), 400

    if not allowed_file(image_file.filename):
        return jsonify({"error": f"File type not allowed. Allowed types are: {', '.join(ALLOWED_EXTENSIONS)}. Got: {image_file.filename}"}), 400

    try:
        pil_image = Image.open(image_file.stream)
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')

        img_byte_arr = io.BytesIO()
        pil_image.save(img_byte_arr, format='JPEG')
        image_bytes = img_byte_arr.getvalue()

        # MIME type for JPEG
        image_part = {"mime_type": "image/jpeg", "data": image_bytes}

    except Exception as e:
        print(f"Error processing image file: {str(e)}")
        return jsonify({"error": f"Error processing image file: {str(e)}"}), 500

    prompt = """You are an expert medical prescription analyzer. Analyze this prescription image and extract the following information:
1. All medications listed in the prescription
2. The dosage for each medication
3. The medical condition or purpose of each medication
4. Any special instructions

Format your response as a structured JSON with the following format:
{
  "patient_info": {
    "name": "",
    "date": ""
  },
  "medications": [
    {
      "medicine": "Medication Name 1",
      "dosage": "Dosage information",
      "purpose": "Used for treating condition",
      "instructions": "Special instructions if any"
    },
    {
      "medicine": "Medication Name 2",
      "dosage": "Dosage information",
      "purpose": "Used for treating condition",
      "instructions": "Special instructions if any"
    }
  ],
  "additional_instructions": "",
  "doctor_info": {
    "name": "",
    "contact": ""
  }
}

Be sure to extract accurate information and maintain the precise JSON format. If you can\'t determine certain information, use empty strings or "Unknown" as values.
"""

    gemini_response_text = ""
    try:
        response = gemini_model_client.generate_content(
            [prompt, image_part])  # Send prompt and image
        gemini_response_text = response.text.strip()

    except Exception as e:
        print(f"Error during Gemini API call: {e}")
        return jsonify({"error": "Error processing request with Gemini model", "details": str(e)}), 500

    if not gemini_response_text:
        return jsonify({"error": "Gemini model returned no decipherable text from the image."}), 500

    try:
        # Attempt to extract JSON from potentially mixed response (e.g. if model adds ```json ... ```)
        json_start = gemini_response_text.find('{')
        json_end = gemini_response_text.rfind('}') + 1
        if json_start != -1 and json_end > json_start:
            json_str = gemini_response_text[json_start:json_end]
            parsed_data = json.loads(json_str)
        else:
            # If no clear JSON block, try to parse the whole string.
            # This might fail if the model's response isn't pure JSON.
            parsed_data = json.loads(gemini_response_text)

    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON response from Gemini: {e}")
        print(f"Raw Gemini response: {gemini_response_text}")
        return jsonify({
            "error": "Failed to parse JSON response from Gemini model.",
            "details": str(e),
            "raw_model_output": gemini_response_text
        }), 500
    except Exception as e:  # Catch any other unexpected errors during parsing
        print(
            f"An unexpected error occurred while parsing Gemini response: {e}")
        print(f"Raw Gemini response: {gemini_response_text}")
        return jsonify({
            "error": "An unexpected error occurred while parsing the Gemini response.",
            "details": str(e),
            "raw_model_output": gemini_response_text
        }), 500

    # Basic check for expected structure
    if not isinstance(parsed_data, dict) or "medications" not in parsed_data:
        print(
            f"Parsed data does not have expected structure. Data: {parsed_data}")
        return jsonify({
            "warning": "Gemini: Parsed data does not have the expected structure (e.g., missing 'medications' key).",
            "raw_model_output": gemini_response_text,
            "parsed_data": parsed_data
        }), 200  # Return 200 with warning, as some data might still be useful

    return jsonify(parsed_data)


@app.route("/food_review", methods=["POST"])
def food_review():
    # Check if an image file is provided
    if 'image' not in request.files and len(request.files) == 0:
        return jsonify({"error": "No image file provided"}), 400

    # Get the image file
    image_file = None
    if 'image' in request.files:
        image_file = request.files['image']
    elif len(request.files) == 1:
        image_file = next(iter(request.files.values()))

    if not image_file or not image_file.filename:
        return jsonify({"error": "Image file is invalid or empty"}), 400

    if not allowed_file(image_file.filename):
        return jsonify({"error": f"File type not allowed. Allowed types are: {', '.join(ALLOWED_EXTENSIONS)}"}), 400

    try:
        # Load food dataset
        food_df = pd.read_csv("food_dataset.csv")

        # Create food embeddings if not already created
        if not hasattr(app, 'food_vectorstore'):
            # Create food descriptions for embedding
            food_descriptions = []
            for _, row in food_df.iterrows():
                desc = f"Food name: {row['Food Name']}, State: {row['State']}, Diet type: {row['Diet Type']}"
                food_descriptions.append(desc)

            # Create food metadata
            food_metadata = [
                {
                    "id": row['id'],
                    "name": row['Food Name'],
                    "state": row['State'],
                    "diet_type": row['Diet Type'],
                    "calories": row['Calories kcal'],
                    "protein": row['Protein g'],
                    "iron": row['Iron mg'],
                    "calcium": row['Calcium mg'],
                    "folic_acid": row['Folic Acid mcg'],
                    "vitamin_c": row['Vitamin C mg'],
                    "vitamin_d": row['Vitamin D IU'],
                    "omega3": row['Omega 3 mg'],
                    "fiber": row['Fiber g']
                } for _, row in food_df.iterrows()
            ]

            # Initialize food vector store
            food_embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2")
            app.food_vectorstore = FAISS.from_texts(
                food_descriptions, food_embeddings, metadatas=food_metadata)
            print("Food vector database created successfully")

        # No OCR - use Gemini directly for food identification
        if not gemini_model_client:
            return jsonify({"error": "Gemini API not configured. Required for food recognition."}), 503

        # Process the image with Gemini
        try:
            # Convert image for Gemini
            pil_image = Image.open(image_file.stream)
            if pil_image.mode not in ['RGB', 'RGBA']:
                pil_image = pil_image.convert('RGB')

            img_byte_arr = io.BytesIO()
            pil_image.save(img_byte_arr, format='JPEG')
            image_bytes = img_byte_arr.getvalue()
            image_part = {"mime_type": "image/jpeg", "data": image_bytes}

            # Prompt for Gemini
            gemini_prompt = """
            You are a food recognition expert. Analyze this image and identify all food items visible.
            Return only a JSON object with numbered food items like this example:
            {
                "food": {
                    "1": "rice",
                    "2": "chapati",
                    "3": "palak paneer",
                    "4": "dal"
                }
            }
            Be specific with the food names and only include Indian food items visible in the image.
            """

            response = gemini_model_client.generate_content(
                [gemini_prompt, image_part])
            gemini_text = response.text.strip()

            # Extract JSON from response
            json_start = gemini_text.find('{')
            json_end = gemini_text.rfind('}') + 1

            if json_start != -1 and json_end > json_start:
                json_str = gemini_text[json_start:json_end]
                parsed_data = json.loads(json_str)
                if "food" in parsed_data and isinstance(parsed_data["food"], dict):
                    food_items = parsed_data["food"]
                else:
                    return jsonify({"error": "Gemini didn't return food items in expected format"}), 400
            else:
                return jsonify({"error": "Couldn't parse JSON from Gemini response",
                                "raw_response": gemini_text}), 400

        except Exception as e:
            return jsonify({"error": f"Error processing image with Gemini: {str(e)}"}), 500

        # RAG process for each food item
        result = []
        total_calories = 0
        total_protein = 0

        for item_id, food_name in food_items.items():
            # Search for the food item in the vector database
            query = f"Food name: {food_name}"
            retrieved_docs = app.food_vectorstore.similarity_search(query, k=5) # Increased k

            # Initialize food_info with base data
            food_info = {
                "name": food_name,
                "id": item_id
            }

            exact_match_found = False
            if retrieved_docs:
                # Try to find an exact match (case-insensitive) in the retrieved results
                for doc in retrieved_docs:
                    doc_name_lower = doc.metadata.get("name", "").lower()
                    if food_name.lower() == doc_name_lower:
                        best_match = doc.metadata
                        exact_match_found = True
                        print(f"Exact match found for '{food_name}': {doc.metadata.get('name')}")
                        break
                
                # If no exact match found, fall back to the top semantic match
                if not exact_match_found:
                    best_match = retrieved_docs[0].metadata
                    print(f"No exact match for '{food_name}', using top semantic match: {best_match.get('name')}")
                
                food_info.update({
                    "calories": best_match.get("calories"),
                    "protein": best_match.get("protein"),
                    "iron": best_match.get("iron"),
                    "calcium": best_match.get("calcium"),
                    "folic_acid": best_match.get("folic_acid"),
                    "vitamin_c": best_match.get("vitamin_c"),
                    "matched_to": best_match.get("name"),
                    "state": best_match.get("state")
                })

                # Update totals
                total_calories += float(best_match.get("calories", 0))
                total_protein += float(best_match.get("protein", 0))
            else:
                # Use Gemma to estimate values if no match found
                estimation_prompt = f"""
                As a nutrition expert, estimate these nutritional values for {food_name}:
                - Calories (kcal)
                - Protein (g)
                - Iron (mg)
                - Calcium (mg)
                - Folic Acid (mcg)
                
                Format your response as a JSON object:
                {{
                    "calories": 150,
                    "protein": 4.5,
                    "iron": 0.8,
                    "calcium": 30,
                    "folic_acid": 25
                }}
                """

                gemma_response = gemma(estimation_prompt, max_new_tokens=200,
                                       truncation=True, return_full_text=False)

                if gemma_response and isinstance(gemma_response, list) and gemma_response[0].get("generated_text"):
                    response_text = gemma_response[0]["generated_text"].strip()

                    # Extract JSON from response
                    json_start = response_text.find('{')
                    json_end = response_text.rfind('}') + 1
                    if json_start != -1 and json_end > json_start:
                        json_str = response_text[json_start:json_end]
                        try:
                            nutrition_data = json.loads(json_str)
                            food_info.update({
                                "calories": nutrition_data.get("calories", 100),
                                "protein": nutrition_data.get("protein", 2),
                                "iron": nutrition_data.get("iron", 0.5),
                                "calcium": nutrition_data.get("calcium", 20),
                                "folic_acid": nutrition_data.get("folic_acid", 15),
                                "estimated": True
                            })

                            # Update totals
                            total_calories += float(
                                nutrition_data.get("calories", 100))
                            total_protein += float(nutrition_data.get("protein", 2))
                        except json.JSONDecodeError:
                            # Default values if JSON parsing fails
                            food_info.update({
                                "calories": 100,
                                "protein": 2,
                                "iron": 0.5,
                                "calcium": 20,
                                "folic_acid": 15,
                                "estimated": True
                            })

                            # Update totals with defaults
                            total_calories += 100
                            total_protein += 2

            result.append(food_info)

        # Generate meal review using Gemma
        food_names_str = ", ".join([f"{item['name']}" for item in result])
        review_prompt = f"""
        As a nutritionist, provide a brief review of this meal consisting of: {food_names_str}.
        Total calories: {total_calories:.1f} kcal, Total protein: {total_protein:.1f}g.
        
        Keep your response to 2-3 sentences focusing on nutritional value, balance, and health benefits.
        """
        
        meal_review = ""  # Initialize to empty string
        try:
            review_response = gemma(
                review_prompt, max_new_tokens=150, truncation=True, return_full_text=False)
            if review_response and isinstance(review_response, list) and review_response[0].get("generated_text"):
                meal_review = review_response[0]["generated_text"].strip()
            elif review_response: # Log if response format is unexpected but not an exception
                print(f"Gemma review generation returned unexpected format: {review_response}")
            else: # Log if response is None or empty
                print("Gemma review generation failed or returned None/empty response.")
        except Exception as e:
            print(f"Error during Gemma meal review generation: {str(e)}")
            # meal_review remains "" as initialized

        return jsonify({
            "items": result,
            "total_nutrition": {
                "calories": round(total_calories, 1),
                "protein": round(total_protein, 1)
            },
            "review": meal_review
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Error processing request: {str(e)}"}), 500


@app.route("/gemini_food_review", methods=["POST"])
def gemini_food_review_endpoint():
    # Check if an image file is provided
    if 'image' not in request.files and len(request.files) == 0:
        return jsonify({"error": "No image file provided"}), 400

    # Get the image file
    image_file = None
    if 'image' in request.files:
        image_file = request.files['image']
    elif len(request.files) == 1:
        image_file = next(iter(request.files.values()))

    if not image_file or not image_file.filename:
        return jsonify({"error": "Image file is invalid or empty"}), 400

    if not allowed_file(image_file.filename):
        return jsonify({"error": f"File type not allowed. Allowed types are: {', '.join(ALLOWED_EXTENSIONS)}"}), 400

    try:
        # Load food dataset
        food_df = pd.read_csv("food_dataset.csv")

        # Create food embeddings if not already created
        if not hasattr(app, 'food_vectorstore'):
            # Create food descriptions for embedding
            food_descriptions = []
            for _, row in food_df.iterrows():
                desc = f"Food name: {row['Food Name']}, State: {row['State']}, Diet type: {row['Diet Type']}"
                food_descriptions.append(desc)

            # Create food metadata
            food_metadata = [
                {
                    "id": row['id'],
                    "name": row['Food Name'],
                    "state": row['State'],
                    "diet_type": row['Diet Type'],
                    "calories": row['Calories kcal'],
                    "protein": row['Protein g'],
                    "iron": row['Iron mg'],
                    "calcium": row['Calcium mg'],
                    "folic_acid": row['Folic Acid mcg'],
                    "vitamin_c": row['Vitamin C mg'],
                    "vitamin_d": row['Vitamin D IU'],
                    "omega3": row['Omega 3 mg'],
                    "fiber": row['Fiber g']
                } for _, row in food_df.iterrows()
            ]

            # Initialize food vector store
            food_embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2")
            app.food_vectorstore = FAISS.from_texts(
                food_descriptions, food_embeddings, metadatas=food_metadata)
            print("Food vector database created successfully")

        # No OCR - use Gemini 1.5 Flash specifically for food identification
        # Instantiate Gemini 1.5 Flash model specifically for this endpoint
        try:
            gemini_1_5_flash_model = genai.GenerativeModel('gemini-1.5-flash') # Explicitly use gemini-1.5-flash
        except Exception as e:
            print(f"Error instantiating Gemini 1.5 Flash model: {e}")
            return jsonify({"error": "Could not initialize Gemini 1.5 Flash model. Check model name and API key permissions."}), 503

        # Process the image with Gemini 1.5 Flash
        try:
            # Convert image for Gemini
            pil_image = Image.open(image_file.stream)
            if pil_image.mode not in ['RGB', 'RGBA']:
                pil_image = pil_image.convert('RGB')

            img_byte_arr = io.BytesIO()
            pil_image.save(img_byte_arr, format='JPEG')
            image_bytes = img_byte_arr.getvalue()
            image_part = {"mime_type": "image/jpeg", "data": image_bytes}

            # Prompt for Gemini (same as in original /food_review)
            gemini_prompt = """
            You are a food recognition expert. Analyze this image and identify all food items visible.
            Return only a JSON object with numbered food items like this example:
            {
                "food": {
                    "1": "rice",
                    "2": "chapati",
                    "3": "palak paneer",
                    "4": "dal"
                }
            }
            Be specific with the food names and only include Indian food items visible in the image.
            """

            response = gemini_1_5_flash_model.generate_content( # Use the specific gemini-1.5-flash model instance
                [gemini_prompt, image_part])
            gemini_text = response.text.strip()

            # Extract JSON from response
            json_start = gemini_text.find('{')
            json_end = gemini_text.rfind('}') + 1

            if json_start != -1 and json_end > json_start:
                json_str = gemini_text[json_start:json_end]
                parsed_data = json.loads(json_str)
                if "food" in parsed_data and isinstance(parsed_data["food"], dict):
                    food_items = parsed_data["food"]
                else:
                    return jsonify({"error": "Gemini 1.5 Flash didn't return food items in expected format"}), 400
            else:
                return jsonify({"error": "Couldn't parse JSON from Gemini 1.5 Flash response",
                                "raw_response": gemini_text}), 400

        except Exception as e:
            return jsonify({"error": f"Error processing image with Gemini 1.5 Flash: {str(e)}"}), 500

        # RAG process for each food item (identical to /food_review)
        result = []
        total_calories = 0
        total_protein = 0

        for item_id, food_name in food_items.items():
            query = f"Food name: {food_name}"
            retrieved_docs = app.food_vectorstore.similarity_search(query, k=5)
            food_info = {"name": food_name, "id": item_id}
            exact_match_found = False
            if retrieved_docs:
                for doc in retrieved_docs:
                    doc_name_lower = doc.metadata.get("name", "").lower()
                    if food_name.lower() == doc_name_lower:
                        best_match = doc.metadata
                        exact_match_found = True
                        print(f"Exact match found for '{food_name}': {doc.metadata.get('name')}")
                        break
                if not exact_match_found:
                    best_match = retrieved_docs[0].metadata
                    print(f"No exact match for '{food_name}', using top semantic match: {best_match.get('name')}")
                
                food_info.update({
                    "calories": best_match.get("calories"),
                    "protein": best_match.get("protein"),
                    "iron": best_match.get("iron"),
                    "calcium": best_match.get("calcium"),
                    "folic_acid": best_match.get("folic_acid"),
                    "vitamin_c": best_match.get("vitamin_c"),
                    "matched_to": best_match.get("name"),
                    "state": best_match.get("state")
                })
                total_calories += float(best_match.get("calories", 0))
                total_protein += float(best_match.get("protein", 0))
            else:
                estimation_prompt = f"""
                As a nutrition expert, estimate these nutritional values for {food_name}:
                - Calories (kcal)
                - Protein (g)
                - Iron (mg)
                - Calcium (mg)
                - Folic Acid (mcg)
                
                Format your response as a JSON object:
                {{
                    "calories": 150,
                    "protein": 4.5,
                    "iron": 0.8,
                    "calcium": 30,
                    "folic_acid": 25
                }}
                """
                gemma_response = gemma(estimation_prompt, max_new_tokens=200,
                                       truncation=True, return_full_text=False)
                if gemma_response and isinstance(gemma_response, list) and gemma_response[0].get("generated_text"):
                    response_text = gemma_response[0]["generated_text"].strip()
                    json_start_gemma = response_text.find('{')
                    json_end_gemma = response_text.rfind('}') + 1
                    if json_start_gemma != -1 and json_end_gemma > json_start_gemma:
                        json_str_gemma = response_text[json_start_gemma:json_end_gemma]
                        try:
                            nutrition_data = json.loads(json_str_gemma)
                            food_info.update({
                                "calories": nutrition_data.get("calories", 100),
                                "protein": nutrition_data.get("protein", 2),
                                "iron": nutrition_data.get("iron", 0.5),
                                "calcium": nutrition_data.get("calcium", 20),
                                "folic_acid": nutrition_data.get("folic_acid", 15),
                                "estimated": True
                            })
                            total_calories += float(nutrition_data.get("calories", 100))
                            total_protein += float(nutrition_data.get("protein", 2))
                        except json.JSONDecodeError:
                            food_info.update({"calories": 100, "protein": 2, "estimated": True, "error": "Gemma estimation JSON parse failed"})
                            total_calories += 100
                            total_protein += 2
                    else:
                        food_info.update({"calories": 100, "protein": 2, "estimated": True, "error": "Gemma estimation no JSON found"})
                        total_calories += 100
                        total_protein += 2
                else:
                    food_info.update({"calories": 100, "protein": 2, "estimated": True, "error": "Gemma estimation failed"})
                    total_calories += 100
                    total_protein += 2
            result.append(food_info)

        # Generate meal review using Gemma (identical to /food_review)
        food_names_str = ", ".join([f"{item['name']}" for item in result])
        review_prompt = f"""
        As a nutritionist, provide a brief review of this meal consisting of: {food_names_str}.
        Total calories: {total_calories:.1f} kcal, Total protein: {total_protein:.1f}g.
        
        Keep your response to 2-3 sentences focusing on nutritional value, balance, and health benefits.
        """
        
        meal_review = ""  # Initialize to empty string
        try:
            review_response = gemma(
                review_prompt, max_new_tokens=150, truncation=True, return_full_text=False)
            if review_response and isinstance(review_response, list) and review_response[0].get("generated_text"):
                meal_review = review_response[0]["generated_text"].strip()
            elif review_response: # Log if response format is unexpected but not an exception
                print(f"Gemma review generation returned unexpected format: {review_response}")
            else: # Log if response is None or empty
                print("Gemma review generation failed or returned None/empty response.")
        except Exception as e:
            print(f"Error during Gemma meal review generation: {str(e)}")
            # meal_review remains "" as initialized

        return jsonify({
            "items": result,
            "total_nutrition": {
                "calories": round(total_calories, 1),
                "protein": round(total_protein, 1)
            },
            "review": meal_review
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Error processing request: {str(e)}"}), 500


@app.route("/week", methods=["POST"])
def get_week_info():
    data = request.get_json()
    week = data.get("week")
    
    if not week:
        return jsonify({"error": "Missing parameter: week"}), 400
    
    try:
        week = int(week)
        if week < 1 or week > 40:
            return jsonify({"error": "Week must be between 1 and 40"}), 400
    except ValueError:
        return jsonify({"error": "Week must be an integer"}), 400
    
    try:
        # Load timeline dataset with proper encoding handling
        if not hasattr(app, 'timeline_df'):
            try:
                # Try different encodings if UTF-8 fails
                app.timeline_df = pd.read_csv("timeline_dataset.csv", encoding='utf-8')
            except UnicodeDecodeError:
                try:
                    # Try Latin-1 encoding which is more permissive
                    app.timeline_df = pd.read_csv("timeline_dataset.csv", encoding='latin1')
                except Exception as e:
                    # Fall back to an even more permissive encoding that replaces problematic characters
                    app.timeline_df = pd.read_csv("timeline_dataset.csv", encoding='cp1252', 
                                                 error_bad_lines=False, warn_bad_lines=True)
            
            # Create timeline vector store if not already created
            if not hasattr(app, 'timeline_vectorstore'):
                # Create descriptions for embedding
                timeline_descriptions = []
                timeline_metadata = []
                
                for idx, row in app.timeline_df.iterrows():
                    desc = f"Week {row['Week']}: Problem: {row['Problem']}, Solution: {row['Solution']}"
                    timeline_descriptions.append(desc)
                    timeline_metadata.append({
                        "week": row['Week'],
                        "problem": row['Problem'],
                        "solution": row['Solution'],
                        "row_index": idx
                    })
                
                # Initialize timeline vector store
                if not hasattr(app, 'embeddings'):
                    app.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                
                app.timeline_vectorstore = FAISS.from_texts(
                    timeline_descriptions, app.embeddings, metadatas=timeline_metadata)
                print("Timeline vector database created successfully")
        
        # Filter dataset directly for the specific week
        week_data = app.timeline_df[app.timeline_df['Week'] == week]
        
        # If direct filtering doesn't return results, use vector search as backup
        if week_data.empty:
            query = f"Pregnancy information for week {week}"
            results = app.timeline_vectorstore.similarity_search(query, k=3)
            
            # Extract data from results
            problems_solutions = []
            for doc in results:
                metadata = doc.metadata
                # Only include if week matches exactly
                if metadata.get('week') == week:
                    problems_solutions.append({
                        "problem": metadata.get('problem'),
                        "solution": metadata.get('solution')
                    })
        else:
            # Use direct filtering results
            problems_solutions = []
            for _, row in week_data.iterrows():
                problems_solutions.append({
                    "problem": row['Problem'],
                    "solution": row['Solution']
                })
        
        # Use Gemma to generate fetus size and weight information
        size_weight_prompt = f"""
        Provide factual information about fetal development during week {week} of pregnancy.
        Include only:
        1. Size comparison (e.g., "size of a lemon")
        2. Approximate weight in grams and ounces
        3. One key developmental milestone for this week
        
        Format your response as a JSON object:
        {{
            "size": "size comparison",
            "weight_grams": number,
            "weight_oz": number,
            "development": "brief description of key development"
        }}
        
        Be accurate, concise, and only include the JSON with no other text.
        """
        
        gemma_response = gemma(size_weight_prompt, max_new_tokens=300, 
                              truncation=True, return_full_text=False)
        
        fetal_info = {}
        if gemma_response and isinstance(gemma_response, list) and gemma_response[0].get("generated_text"):
            response_text = gemma_response[0]["generated_text"].strip()
            
            # Extract JSON from response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                try:
                    fetal_info = json.loads(json_str)
                except json.JSONDecodeError:
                    # If parsing fails, create default info
                    fetal_info = {
                        "size": "Information not available",
                        "weight_grams": 0,
                        "weight_oz": 0,
                        "development": "Information not available"
                    }
        else:
            fetal_info = {
                "size": "Information not available",
                "weight_grams": 0, 
                "weight_oz": 0,
                "development": "Information not available"
            }
        
        # Construct response
        response = {
            "week": week,
            "fetal_development": fetal_info,
            "common_issues": problems_solutions
        }
        
        return jsonify(response)
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Error processing request: {str(e)}"}), 500


@app.route("/gemini_chatbot", methods=["POST"])
def gemini_chatbot_endpoint():
    data = request.get_json(silent=True) # Expects JSON
    
    # This check ensures the data is JSON and has the 'text' field
    if not data or not isinstance(data, dict) or "text" not in data or not data["text"].strip():
        return jsonify({"error": "Invalid or missing JSON payload. Please send a JSON object with a non-empty 'text' field and ensure Content-Type is 'application/json'"}), 400

    user_text = data["text"]

    try:
        system_prompt = "You are a pregnancy assistance that will answer all the question related to pregnancy and strictly adhere to the prompt user gives no fooling around give no other suggestions than pregnancy  "
        
        chatbot_model = genai.GenerativeModel(
            'gemini-1.5-flash',
            system_instruction=system_prompt
        )
        
        # For a simple text-in, text-out chat, you can directly send the content.
        response = chatbot_model.generate_content(user_text)

        # Extract the text response from the model's output
        chatbot_response = ""
        if response.parts:
            chatbot_response = response.parts[0].text
        elif response.candidates:
            chatbot_response = response.candidates[0].content.parts[0].text
        else:
            chatbot_response = response.text

        return jsonify({"response": chatbot_response})

    except genai.types.generation_types.BlockedPromptException:
        return jsonify({"error": "Prompt blocked by safety filters"}), 400
    except Exception as e:
        import traceback
        traceback.print_exc()
        error_message = str(e)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
