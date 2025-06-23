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
import time  # Added for performance tracking
import datetime  # Added for timestamp tracking
from collections import defaultdict  # Added for statistics tracking

from dotenv import load_dotenv

load_dotenv()

# --- AI Response Quality Analytics System ---
ai_response_analytics = {
    "total_responses": 0,
    "hallucination_detection": {
        "total_analyzed": 0,
        "hallucinated_responses": 0,
        "accuracy_score": 0.0,
        "csv_data_usage_rate": 0.0
    },
    "semantic_accuracy": {
        "exercise_name_accuracy": 0.0,
        "week_accuracy": 0.0,
        "recommendation_accuracy": 0.0,
        "safety_info_accuracy": 0.0
    },
    "response_quality": {
        "structured_responses": 0,
        "malformed_responses": 0,
        "complete_responses": 0,
        "missing_fields": []
    },
    "dataset_utilization": {
        "csv_exact_match_rate": 0.0,
        "rag_fallback_rate": 0.0,
        "no_data_responses": 0,
        "fields_used_correctly": {}
    },
    "model_performance": {
        "gemini_accuracy": 0.0,
        "gemma_accuracy": 0.0,
        "response_consistency": 0.0
    },
    "detailed_analysis": []
}

def analyze_ai_response_quality(user_input, ai_response, csv_data_used, model_type, endpoint):
    """Analyze AI response for hallucination, accuracy, and quality"""
    global ai_response_analytics
    
    analysis = {
        "timestamp": datetime.datetime.now().isoformat(),
        "endpoint": endpoint,
        "model_type": model_type,
        "user_input": user_input,
        "csv_data_found": bool(csv_data_used),
        "hallucination_score": 0.0,
        "accuracy_score": 0.0,
        "quality_score": 0.0,
        "issues_detected": []
    }
    
    # 1. HALLUCINATION DETECTION
    hallucination_score = detect_hallucination(ai_response, csv_data_used, user_input)
    analysis["hallucination_score"] = hallucination_score
    
    # 2. ACCURACY ANALYSIS
    accuracy_score = measure_response_accuracy(ai_response, csv_data_used, user_input)
    analysis["accuracy_score"] = accuracy_score
    
    # 3. SEMANTIC CORRECTNESS
    semantic_score = check_semantic_correctness(ai_response, csv_data_used, user_input)
    analysis["semantic_score"] = semantic_score
    
    # 4. RESPONSE STRUCTURE QUALITY
    structure_score = validate_response_structure(ai_response)
    analysis["structure_score"] = structure_score
    
    # 5. CSV DATA USAGE ANALYSIS
    csv_usage_score = analyze_csv_data_usage(ai_response, csv_data_used)
    analysis["csv_usage_score"] = csv_usage_score
    
    # Calculate overall quality score
    analysis["quality_score"] = (
        hallucination_score * 0.3 + 
        accuracy_score * 0.25 + 
        semantic_score * 0.2 + 
        structure_score * 0.15 + 
        csv_usage_score * 0.1
    )
    
    # Update global analytics
    update_ai_analytics(analysis)
    
    # Store detailed analysis (keep last 500)
    ai_response_analytics["detailed_analysis"].append(analysis)
    if len(ai_response_analytics["detailed_analysis"]) > 500:
        ai_response_analytics["detailed_analysis"] = ai_response_analytics["detailed_analysis"][-500:]
    
    return analysis

def detect_hallucination(ai_response, csv_data, user_input):
    """Detect if AI is hallucinating information not in CSV data"""
    hallucination_score = 1.0  # Start with perfect score
    
    try:
        # If CSV data exists, check if AI response contradicts it
        if csv_data:
            # Check if AI mentions data not in CSV
            csv_fields = ['name', 'N', 'benefits', 'contraindications', 'modifications', 
                         'intensity', 'rest_interval', 'equipment_needed', 'primary_muscles',
                         'safety_tips', 'fitness_level', 'medical_clearance', 
                         'progression_guidelines', 'postpartum_relevance']
            
            response_text = str(ai_response).lower()
            
            # Check for specific CSV field values
            for field in csv_fields:
                csv_value = str(csv_data.get(field, '')).lower()
                if csv_value and csv_value != 'nan' and len(csv_value) > 2:
                    # If AI contradicts CSV data
                    if csv_value in response_text:
                        continue  # Good, using CSV data
                    else:
                        # Check if AI is making up different information
                        if field == 'N' and 'sets' in response_text or 'reps' in response_text:
                            # Look for number contradictions
                            import re
                            numbers = re.findall(r'\d+', response_text)
                            csv_number = str(csv_data.get('N', ''))
                            if csv_number and csv_number not in numbers:
                                hallucination_score -= 0.2
            
            # Check for made-up exercise names
            csv_exercise_name = str(csv_data.get('name', '')).lower()
            if csv_exercise_name and csv_exercise_name not in response_text:
                hallucination_score -= 0.3
                
        else:
            # No CSV data found - check if AI is making specific claims
            response_text = str(ai_response).lower()
            suspicious_phrases = [
                'studies show', 'research indicates', 'according to experts',
                'clinical trials', 'medical guidelines recommend'
            ]
            for phrase in suspicious_phrases:
                if phrase in response_text:
                    hallucination_score -= 0.1
    
    except Exception as e:
        hallucination_score = 0.5  # Neutral score on error
    
    return max(0.0, min(1.0, hallucination_score))

def measure_response_accuracy(ai_response, csv_data, user_input):
    """Measure how accurate the AI response is compared to CSV data"""
    accuracy_score = 0.0
    
    try:
        if not csv_data:
            return 0.3  # Low score if no CSV data used
        
        response_text = str(ai_response).lower()
        correct_matches = 0
        total_checkable = 0
        
        # Check key fields for accuracy
        checkable_fields = {
            'name': user_input.get('name', ''),
            'N': 'sets',
            'benefits': 'benefit',
            'intensity': 'intensity',
            'primary_muscles': 'muscle',
            'equipment_needed': 'equipment'
        }
        
        for field, keyword in checkable_fields.items():
            csv_value = str(csv_data.get(field, '')).lower()
            if csv_value and csv_value != 'nan' and len(csv_value) > 2:
                total_checkable += 1
                if keyword in response_text and csv_value in response_text:
                    correct_matches += 1
        
        if total_checkable > 0:
            accuracy_score = correct_matches / total_checkable
        else:
            accuracy_score = 0.5
            
    except Exception as e:
        accuracy_score = 0.5
    
    return max(0.0, min(1.0, accuracy_score))

def check_semantic_correctness(ai_response, csv_data, user_input):
    """Check semantic correctness of AI response"""
    semantic_score = 1.0
    
    try:
        response_text = str(ai_response).lower()
        
        # Check for logical consistency
        if 'week_pregnancy' in user_input:
            week = user_input['week_pregnancy']
            
            # Check trimester consistency
            if week <= 12 and '2nd trimester' in response_text:
                semantic_score -= 0.3
            elif week > 12 and week <= 27 and '3rd trimester' in response_text:
                semantic_score -= 0.3
            elif week > 27 and '1st trimester' in response_text:
                semantic_score -= 0.3
        
        # Check for contradictory statements
        contradictions = [
            ('low intensity', 'high intensity'),
            ('not recommended', 'highly recommended'),
            ('avoid', 'perform regularly')
        ]
        
        for contradiction in contradictions:
            if contradiction[0] in response_text and contradiction[1] in response_text:
                semantic_score -= 0.2
                
    except Exception as e:
        semantic_score = 0.5
    
    return max(0.0, min(1.0, semantic_score))

def validate_response_structure(ai_response):
    """Validate if response follows expected structure"""
    structure_score = 0.0
    
    try:
        if isinstance(ai_response, dict):
            # JSON response structure validation
            expected_sections = [
                'exercise_analysis', 'technical_details', 'safety_guidelines',
                'benefits_and_progression', 'recommendations', 'summary'
            ]
            
            present_sections = sum(1 for section in expected_sections if section in ai_response)
            structure_score = present_sections / len(expected_sections)
            
        else:
            # Text response quality check
            response_text = str(ai_response)
            if len(response_text) > 50:
                structure_score = 0.7  # Basic text response
            if len(response_text) > 200:
                structure_score = 0.8  # Detailed text response
                
    except Exception as e:
        structure_score = 0.1
    
    return max(0.0, min(1.0, structure_score))

def analyze_csv_data_usage(ai_response, csv_data):
    """Analyze how well AI uses CSV data"""
    usage_score = 0.0
    
    try:
        if not csv_data:
            return 0.0
        
        response_text = str(ai_response).lower()
        csv_fields_used = 0
        total_available_fields = 0
        
        important_fields = [
            'name', 'N', 'benefits', 'contraindications', 'modifications',
            'safety_tips', 'equipment_needed', 'primary_muscles'
        ]
        
        for field in important_fields:
            csv_value = str(csv_data.get(field, '')).lower()
            if csv_value and csv_value != 'nan' and len(csv_value) > 2:
                total_available_fields += 1
                # Check if this CSV data appears in AI response
                if csv_value in response_text or any(word in response_text for word in csv_value.split()):
                    csv_fields_used += 1
        
        if total_available_fields > 0:
            usage_score = csv_fields_used / total_available_fields
        
    except Exception as e:
        usage_score = 0.0
    
    return max(0.0, min(1.0, usage_score))

def update_ai_analytics(analysis):
    """Update global AI analytics with new analysis"""
    global ai_response_analytics
    
    ai_response_analytics["total_responses"] += 1
    
    # Update hallucination detection
    ai_response_analytics["hallucination_detection"]["total_analyzed"] += 1
    if analysis["hallucination_score"] < 0.7:
        ai_response_analytics["hallucination_detection"]["hallucinated_responses"] += 1
    
    # Update accuracy (running average)
    current_avg = ai_response_analytics["hallucination_detection"]["accuracy_score"]
    new_score = analysis["accuracy_score"]
    total = ai_response_analytics["total_responses"]
    ai_response_analytics["hallucination_detection"]["accuracy_score"] = (
        (current_avg * (total - 1) + new_score) / total
    )
    
    # Update CSV usage rate
    if analysis["csv_data_found"]:
        current_usage = ai_response_analytics["hallucination_detection"]["csv_data_usage_rate"]
        ai_response_analytics["hallucination_detection"]["csv_data_usage_rate"] = (
            (current_usage * (total - 1) + analysis["csv_usage_score"]) / total
        )
    
    # Update model performance
    model_type = analysis["model_type"]
    if model_type == "gemini":
        current_accuracy = ai_response_analytics["model_performance"]["gemini_accuracy"]
        ai_response_analytics["model_performance"]["gemini_accuracy"] = (
            (current_accuracy * (total - 1) + analysis["quality_score"]) / total
        )
    elif model_type == "gemma":
        current_accuracy = ai_response_analytics["model_performance"]["gemma_accuracy"]
        ai_response_analytics["model_performance"]["gemma_accuracy"] = (
            (current_accuracy * (total - 1) + analysis["quality_score"]) / total
        )

# --- API Statistics and Tracking System ---
api_call_stats = {
    "total_calls": 0,
    "successful_calls": 0,
    "failed_calls": 0,
    "calls_by_endpoint": defaultdict(int),
    "calls_by_hour": defaultdict(int),
    "response_times": [],
    "accuracy_metrics": {
        "exact_csv_matches": 0,
        "rag_fallbacks": 0,
        "semantic_search_accuracy": [],
        "dataset_utilization_rate": 0
    },
    "model_context_usage": {
        "csv_data_used": 0,
        "rag_retrievals": 0,
        "empty_responses": 0,
        "gemini_calls": 0,
        "gemma_calls": 0
    },
    "semantic_search_stats": {
        "total_searches": 0,
        "avg_documents_retrieved": 0,
        "week_match_accuracy": 0,
        "time_match_accuracy": 0,
        "exercise_name_accuracy": 0,
        "similarity_scores": []
    },
    "detailed_calls": []  # Store detailed information about each call
}

def log_api_call(endpoint, request_data, response_data, execution_time, success, error_details=None, search_metrics=None):
    """Log detailed API call information for statistics tracking"""
    global api_call_stats
    
    current_time = datetime.datetime.now()
    hour_key = current_time.strftime("%Y-%m-%d-%H")
    
    # Update basic counters
    api_call_stats["total_calls"] += 1
    api_call_stats["calls_by_endpoint"][endpoint] += 1
    api_call_stats["calls_by_hour"][hour_key] += 1
    api_call_stats["response_times"].append(execution_time)
    
    if success:
        api_call_stats["successful_calls"] += 1
    else:
        api_call_stats["failed_calls"] += 1
    
    # Update model context usage
    if search_metrics:
        if search_metrics.get("exact_match_found"):
            api_call_stats["accuracy_metrics"]["exact_csv_matches"] += 1
            api_call_stats["model_context_usage"]["csv_data_used"] += 1
        elif search_metrics.get("rag_used"):
            api_call_stats["accuracy_metrics"]["rag_fallbacks"] += 1
            api_call_stats["model_context_usage"]["rag_retrievals"] += 1
        
        if search_metrics.get("model_type") == "gemini":
            api_call_stats["model_context_usage"]["gemini_calls"] += 1
        elif search_metrics.get("model_type") == "gemma":
            api_call_stats["model_context_usage"]["gemma_calls"] += 1
        
        # Update semantic search statistics
        if search_metrics.get("documents_retrieved"):
            api_call_stats["semantic_search_stats"]["total_searches"] += 1
            docs_count = search_metrics["documents_retrieved"]
            current_avg = api_call_stats["semantic_search_stats"]["avg_documents_retrieved"]
            total_searches = api_call_stats["semantic_search_stats"]["total_searches"]
            api_call_stats["semantic_search_stats"]["avg_documents_retrieved"] = (
                (current_avg * (total_searches - 1) + docs_count) / total_searches
            )
        
        # Track accuracy metrics
        if search_metrics.get("week_matched"):
            api_call_stats["semantic_search_stats"]["week_match_accuracy"] += 1
        if search_metrics.get("time_matched"):
            api_call_stats["semantic_search_stats"]["time_match_accuracy"] += 1
        if search_metrics.get("exercise_name_matched"):
            api_call_stats["semantic_search_stats"]["exercise_name_accuracy"] += 1
        
        if search_metrics.get("similarity_scores"):
            api_call_stats["semantic_search_stats"]["similarity_scores"].extend(
                search_metrics["similarity_scores"]
            )
    
    # Store detailed call information (keep last 1000 calls)
    call_detail = {
        "timestamp": current_time.isoformat(),
        "endpoint": endpoint,
        "request_data": request_data,
        "response_size": len(str(response_data)) if response_data else 0,
        "execution_time": execution_time,
        "success": success,
        "error_details": error_details,
        "search_metrics": search_metrics
    }
    
    api_call_stats["detailed_calls"].append(call_detail)
    
    # Keep only last 1000 calls to prevent memory issues
    if len(api_call_stats["detailed_calls"]) > 1000:
        api_call_stats["detailed_calls"] = api_call_stats["detailed_calls"][-1000:]

def calculate_accuracy_metrics():
    """Calculate various accuracy and performance metrics"""
    total_calls = api_call_stats["total_calls"]
    if total_calls == 0:
        return {}
    
    # Calculate success rate
    success_rate = (api_call_stats["successful_calls"] / total_calls) * 100
    
    # Calculate dataset utilization rate
    csv_usage = api_call_stats["model_context_usage"]["csv_data_used"]
    rag_usage = api_call_stats["model_context_usage"]["rag_retrievals"]
    dataset_utilization = ((csv_usage + rag_usage) / total_calls) * 100 if total_calls > 0 else 0
    
    # Calculate semantic search accuracy
    total_searches = api_call_stats["semantic_search_stats"]["total_searches"]
    week_accuracy = (api_call_stats["semantic_search_stats"]["week_match_accuracy"] / total_searches * 100) if total_searches > 0 else 0
    time_accuracy = (api_call_stats["semantic_search_stats"]["time_match_accuracy"] / total_searches * 100) if total_searches > 0 else 0
    name_accuracy = (api_call_stats["semantic_search_stats"]["exercise_name_accuracy"] / total_searches * 100) if total_searches > 0 else 0
    
    # Calculate average response time
    avg_response_time = np.mean(api_call_stats["response_times"]) if api_call_stats["response_times"] else 0
    
    # Calculate similarity score statistics
    similarity_scores = api_call_stats["semantic_search_stats"]["similarity_scores"]
    similarity_stats = {
        "avg_similarity": np.mean(similarity_scores) if similarity_scores else 0,
        "min_similarity": np.min(similarity_scores) if similarity_scores else 0,
        "max_similarity": np.max(similarity_scores) if similarity_scores else 0,
        "std_similarity": np.std(similarity_scores) if similarity_scores else 0
    }
    
    return {
        "overall_accuracy": {
            "success_rate": round(success_rate, 2),
            "dataset_utilization_rate": round(dataset_utilization, 2),
            "avg_response_time_ms": round(avg_response_time * 1000, 2)
        },
        "semantic_search_accuracy": {
            "week_match_accuracy": round(week_accuracy, 2),
            "time_match_accuracy": round(time_accuracy, 2),
            "exercise_name_accuracy": round(name_accuracy, 2),
            "total_searches_performed": total_searches,
            "avg_documents_per_search": round(api_call_stats["semantic_search_stats"]["avg_documents_retrieved"], 2)
        },
        "similarity_score_analysis": similarity_stats,
        "model_performance": {
            "exact_csv_matches_percentage": round((csv_usage / total_calls * 100), 2) if total_calls > 0 else 0,
            "rag_fallback_percentage": round((rag_usage / total_calls * 100), 2) if total_calls > 0 else 0,
            "gemini_usage_percentage": round((api_call_stats["model_context_usage"]["gemini_calls"] / total_calls * 100), 2) if total_calls > 0 else 0,
            "gemma_usage_percentage": round((api_call_stats["model_context_usage"]["gemma_calls"] / total_calls * 100), 2) if total_calls > 0 else 0
        }
    }

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


@app.route("/ai_analytics", methods=["GET"])
def ai_analytics_dashboard():
    """AI Response Quality Analytics Dashboard"""
    try:
        # Calculate hallucination rate
        total_analyzed = ai_response_analytics["hallucination_detection"]["total_analyzed"]
        hallucinated = ai_response_analytics["hallucination_detection"]["hallucinated_responses"]
        hallucination_rate = (hallucinated / total_analyzed * 100) if total_analyzed > 0 else 0
        
        # Get recent analysis for trends
        recent_analysis = ai_response_analytics["detailed_analysis"][-50:] if ai_response_analytics["detailed_analysis"] else []
        
        # Calculate score averages
        if recent_analysis:
            avg_hallucination = sum(a["hallucination_score"] for a in recent_analysis) / len(recent_analysis)
            avg_accuracy = sum(a["accuracy_score"] for a in recent_analysis) / len(recent_analysis)
            avg_semantic = sum(a["semantic_score"] for a in recent_analysis) / len(recent_analysis)
            avg_structure = sum(a["structure_score"] for a in recent_analysis) / len(recent_analysis)
            avg_csv_usage = sum(a["csv_usage_score"] for a in recent_analysis) / len(recent_analysis)
            avg_quality = sum(a["quality_score"] for a in recent_analysis) / len(recent_analysis)
        else:
            avg_hallucination = avg_accuracy = avg_semantic = avg_structure = avg_csv_usage = avg_quality = 0.0
        
        # Model comparison
        gemini_responses = [a for a in recent_analysis if a["model_type"] == "gemini"]
        gemma_responses = [a for a in recent_analysis if a["model_type"] == "gemma"]
        
        gemini_avg_quality = sum(r["quality_score"] for r in gemini_responses) / len(gemini_responses) if gemini_responses else 0
        gemma_avg_quality = sum(r["quality_score"] for r in gemma_responses) / len(gemma_responses) if gemma_responses else 0
        
        # CSV data utilization analysis
        csv_used_responses = [a for a in recent_analysis if a["csv_data_found"]]
        csv_utilization_rate = (len(csv_used_responses) / len(recent_analysis) * 100) if recent_analysis else 0
        
        response_data = {
            "dashboard_overview": {
                "total_ai_responses_analyzed": ai_response_analytics["total_responses"],
                "overall_quality_score": round(avg_quality * 100, 1),
                "hallucination_rate_percentage": round(hallucination_rate, 1),
                "csv_data_utilization_rate": round(csv_utilization_rate, 1),
                "last_updated": datetime.datetime.now().isoformat()
            },
            "hallucination_analysis": {
                "detection_accuracy": round(avg_hallucination * 100, 1),
                "total_hallucinations_detected": hallucinated,
                "clean_responses": total_analyzed - hallucinated,
                "hallucination_trends": [
                    {
                        "timestamp": a["timestamp"],
                        "score": round(a["hallucination_score"] * 100, 1),
                        "model": a["model_type"]
                    } for a in recent_analysis[-20:]
                ]
            },
            "accuracy_metrics": {
                "response_accuracy_score": round(avg_accuracy * 100, 1),
                "semantic_correctness_score": round(avg_semantic * 100, 1),
                "structure_quality_score": round(avg_structure * 100, 1),
                "csv_data_usage_score": round(avg_csv_usage * 100, 1)
            },
            "model_performance_comparison": {
                "gemini": {
                    "total_responses": len(gemini_responses),
                    "average_quality_score": round(gemini_avg_quality * 100, 1),
                    "hallucination_rate": round(sum(1 for r in gemini_responses if r["hallucination_score"] < 0.7) / len(gemini_responses) * 100, 1) if gemini_responses else 0
                },
                "gemma": {
                    "total_responses": len(gemma_responses),
                    "average_quality_score": round(gemma_avg_quality * 100, 1),
                    "hallucination_rate": round(sum(1 for r in gemma_responses if r["hallucination_score"] < 0.7) / len(gemma_responses) * 100, 1) if gemma_responses else 0
                }
            },
            "dataset_utilization_analysis": {
                "responses_using_csv_data": len(csv_used_responses),
                "responses_without_csv_data": len(recent_analysis) - len(csv_used_responses),
                "csv_field_usage_breakdown": {
                    "exercise_name_usage": round(sum(1 for a in csv_used_responses if "name" in str(a.get("user_input", {}))) / len(csv_used_responses) * 100, 1) if csv_used_responses else 0,
                    "sets_reps_accuracy": round(sum(1 for a in csv_used_responses if a["accuracy_score"] > 0.8) / len(csv_used_responses) * 100, 1) if csv_used_responses else 0,
                    "safety_info_inclusion": round(sum(1 for a in csv_used_responses if a["csv_usage_score"] > 0.6) / len(csv_used_responses) * 100, 1) if csv_used_responses else 0
                }
            },
            "quality_trends": {
                "last_20_responses": [
                    {
                        "timestamp": a["timestamp"],
                        "overall_quality": round(a["quality_score"] * 100, 1),
                        "hallucination_score": round(a["hallucination_score"] * 100, 1),
                        "accuracy_score": round(a["accuracy_score"] * 100, 1),
                        "model": a["model_type"],
                        "csv_used": a["csv_data_found"]
                    } for a in recent_analysis[-20:]
                ]
            },
            "issues_detected": {
                "common_problems": [
                    "AI making up exercise recommendations not in CSV",
                    "Incorrect trimester information",
                    "Missing safety guidelines from dataset",
                    "Contradictory intensity recommendations"
                ],
                "recent_problem_responses": [
                    {
                        "timestamp": a["timestamp"],
                        "model": a["model_type"],
                        "quality_score": round(a["quality_score"] * 100, 1),
                        "main_issue": "Low hallucination score" if a["hallucination_score"] < 0.7 else "Structure issues" if a["structure_score"] < 0.5 else "CSV data not used"
                    } for a in recent_analysis if a["quality_score"] < 0.6
                ][-10:]
            }
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({
            "error": "Failed to generate AI analytics",
            "details": str(e),
            "basic_stats": {
                "total_responses": ai_response_analytics["total_responses"],
                "analysis_available": len(ai_response_analytics["detailed_analysis"])
            }
        }), 500

@app.route("/ai_dashboard")
def ai_dashboard_ui():
    """AI Analytics Dashboard UI"""
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Response Quality Analytics</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
        .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 10px; margin-bottom: 30px; text-align: center; }
        .header h1 { font-size: 2.5rem; margin-bottom: 10px; }
        .header p { font-size: 1.1rem; opacity: 0.9; }
        .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 30px; }
        .stat-card { background: white; padding: 25px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); text-align: center; }
        .stat-number { font-size: 2.5rem; font-weight: bold; margin-bottom: 5px; }
        .stat-label { color: #666; font-size: 0.9rem; }
        .excellent { color: #27ae60; }
        .good { color: #2980b9; }
        .warning { color: #f39c12; }
        .danger { color: #e74c3c; }
        .chart-container { background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin-bottom: 30px; }
        .chart-title { font-size: 1.5rem; margin-bottom: 20px; color: #333; }
        .model-comparison { display: grid; grid-template-columns: 1fr 1fr; gap: 30px; margin-bottom: 30px; }
        .model-card { background: white; padding: 25px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .model-title { font-size: 1.3rem; margin-bottom: 15px; text-align: center; }
        .model-stat { display: flex; justify-content: space-between; margin-bottom: 10px; }
        .issues-section { background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .issue-item { background: #fff3cd; border: 1px solid #ffeaa7; padding: 15px; border-radius: 5px; margin-bottom: 10px; }
        .refresh-btn { background: #667eea; color: white; border: none; padding: 12px 24px; border-radius: 5px; cursor: pointer; font-size: 1rem; margin: 20px 0; }
        .refresh-btn:hover { background: #5a67d8; }
        .loading { text-align: center; padding: 50px; color: #666; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 AI Response Quality Analytics</h1>
            <p>Real-time monitoring of AI hallucination, accuracy, and dataset utilization</p>
            <div style="margin-top: 20px; background: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px; font-size: 0.9rem;">
                <strong>📊 How We Measure AI Quality:</strong><br>
                • <strong>Hallucination Detection:</strong> We check if AI contradicts our CSV dataset or makes unsupported claims<br>
                • <strong>Accuracy Score:</strong> We verify how many CSV facts the AI correctly uses in responses<br>
                • <strong>Dataset Usage:</strong> We track when AI uses our exercise database vs. making up information
            </div>
        </div>

        <div id="loading" class="loading">
            <h3>Loading AI Analytics...</h3>
        </div>

        <div id="dashboard" style="display: none;">
            <button class="refresh-btn" onclick="loadDashboard()">🔄 Refresh Data</button>
            
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-number excellent" id="qualityScore">--</div>
                    <div class="stat-label">Overall Quality Score</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number" id="hallucinationRate">--</div>
                    <div class="stat-label">Hallucination Rate</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number good" id="csvUsage">--</div>
                    <div class="stat-label">CSV Data Usage</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number" id="totalResponses">--</div>
                    <div class="stat-label">Total Responses Analyzed</div>
                </div>
            </div>

            <div class="chart-container">
                <div class="chart-title">📈 Quality Trends (Last 20 Responses)</div>
                <canvas id="qualityChart" width="400" height="200"></canvas>
            </div>

            <div class="model-comparison">
                <div class="model-card">
                    <div class="model-title">🔮 Gemini Performance</div>
                    <div class="model-stat">
                        <span>Quality Score:</span>
                        <span id="geminiQuality" class="good">--</span>
                    </div>
                    <div class="model-stat">
                        <span>Responses:</span>
                        <span id="geminiResponses">--</span>
                    </div>
                    <div class="model-stat">
                        <span>Hallucination Rate:</span>
                        <span id="geminiHallucination">--</span>
                    </div>
                </div>
                <div class="model-card">
                    <div class="model-title">⚡ Gemma Performance</div>
                    <div class="model-stat">
                        <span>Quality Score:</span>
                        <span id="gemmaQuality" class="good">--</span>
                    </div>
                    <div class="model-stat">
                        <span>Responses:</span>
                        <span id="gemmaResponses">--</span>
                    </div>
                    <div class="model-stat">
                        <span>Hallucination Rate:</span>
                        <span id="gemmaHallucination">--</span>
                    </div>
                </div>
            </div>

            <div class="chart-container">
                <div class="chart-title">🔬 Technical Methodology</div>
                <div style="background: #f8f9fa; padding: 20px; border-radius: 8px; margin-bottom: 20px;">
                    <h4>📋 What to Tell Stakeholders About Our AI Quality Measurement:</h4>
                    
                    <div style="margin: 15px 0;">
                        <strong>1. Hallucination Detection Algorithm:</strong><br>
                        • Starts with 100% trust score<br>
                        • Deducts 30% if AI ignores the correct exercise name from our database<br>
                        • Deducts 20% if AI provides wrong sets/reps numbers<br>
                        • Deducts 10% for each unsupported medical claim (like "studies show...")<br>
                        • <em>Formula: Final Score = 100% - (sum of all deductions)</em>
                    </div>
                    
                    <div style="margin: 15px 0;">
                        <strong>2. Accuracy Measurement:</strong><br>
                        • We check 6 key fields: exercise name, sets, benefits, intensity, muscles, equipment<br>
                        • Count how many CSV facts appear correctly in AI response<br>
                        • <em>Formula: Accuracy = (Correct Facts Used / Total Available Facts) × 100%</em>
                    </div>
                    
                    <div style="margin: 15px 0;">
                        <strong>3. Quality Assurance Process:</strong><br>
                        • Each AI response gets scored on 5 dimensions (0-100%)<br>
                        • Weighted average: Hallucination (30%) + Accuracy (25%) + Semantics (20%) + Structure (15%) + CSV Usage (10%)<br>
                        • All scores stored with timestamps for trend analysis
                    </div>
                    
                    <div style="margin: 15px 0; padding: 10px; background: #e3f2fd; border-left: 4px solid #2196f3;">
                        <strong>💡 Key Insight:</strong> Higher scores mean AI is more reliable and factual. 
                        Scores below 70% indicate potential issues that need investigation.
                    </div>
                </div>
            </div>

            <div class="chart-container">
                <div class="chart-title">📐 Mathematical Breakdown</div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                    <div style="background: #fff3e0; padding: 15px; border-radius: 8px;">
                        <h5>🧮 Hallucination Score Calculation</h5>
                        <code style="background: #f5f5f5; padding: 10px; display: block; margin: 10px 0; border-radius: 4px;">
                            initial_score = 1.0<br>
                            if (wrong_exercise_name): score -= 0.3<br>
                            if (wrong_sets_reps): score -= 0.2<br>
                            for each unsupported_claim: score -= 0.1<br>
                            final_score = max(0, min(1, score))
                        </code>
                    </div>
                    <div style="background: #e8f5e8; padding: 15px; border-radius: 8px;">
                        <h5>📊 Accuracy Score Formula</h5>
                        <code style="background: #f5f5f5; padding: 10px; display: block; margin: 10px 0; border-radius: 4px;">
                            checkable_fields = 6<br>
                            correct_matches = count_matches()<br>
                            accuracy = correct_matches / checkable_fields<br>
                            percentage = accuracy × 100%
                        </code>
                    </div>
                </div>
                <div style="margin-top: 15px; padding: 15px; background: #f0f0f0; border-radius: 8px;">
                    <strong>Example Calculation:</strong><br>
                    If AI response mentions 4 out of 6 available CSV facts correctly → Accuracy = 4/6 = 66.7%<br>
                    If AI uses wrong exercise name → Hallucination = 100% - 30% = 70%<br>
                    Overall Quality = (70% × 0.3) + (66.7% × 0.25) + ... = Combined weighted score
                </div>
            </div>

            <div class="issues-section">
                <div class="chart-title">⚠️ Issues Detected</div>
                <div id="issuesList"></div>
            </div>
        </div>
    </div>

    <script>
        let qualityChart;

        async function loadDashboard() {
            document.getElementById('loading').style.display = 'block';
            document.getElementById('dashboard').style.display = 'none';
            
            try {
                const response = await fetch('/ai_analytics');
                const data = await response.json();
                
                // Update overview stats
                document.getElementById('qualityScore').textContent = data.dashboard_overview.overall_quality_score + '%';
                document.getElementById('hallucinationRate').textContent = data.dashboard_overview.hallucination_rate_percentage + '%';
                document.getElementById('csvUsage').textContent = data.dashboard_overview.csv_data_utilization_rate + '%';
                document.getElementById('totalResponses').textContent = data.dashboard_overview.total_ai_responses_analyzed;
                
                // Color code hallucination rate
                const hallucinationEl = document.getElementById('hallucinationRate');
                const hallucinationRate = data.dashboard_overview.hallucination_rate_percentage;
                if (hallucinationRate < 5) hallucinationEl.className = 'stat-number excellent';
                else if (hallucinationRate < 15) hallucinationEl.className = 'stat-number good';
                else if (hallucinationRate < 30) hallucinationEl.className = 'stat-number warning';
                else hallucinationEl.className = 'stat-number danger';
                
                // Update model comparison
                document.getElementById('geminiQuality').textContent = data.model_performance_comparison.gemini.average_quality_score + '%';
                document.getElementById('geminiResponses').textContent = data.model_performance_comparison.gemini.total_responses;
                document.getElementById('geminiHallucination').textContent = data.model_performance_comparison.gemini.hallucination_rate + '%';
                
                document.getElementById('gemmaQuality').textContent = data.model_performance_comparison.gemma.average_quality_score + '%';
                document.getElementById('gemmaResponses').textContent = data.model_performance_comparison.gemma.total_responses;
                document.getElementById('gemmaHallucination').textContent = data.model_performance_comparison.gemma.hallucination_rate + '%';
                
                // Create quality trend chart
                createQualityChart(data.quality_trends.last_20_responses);
                
                // Update issues list
                updateIssuesList(data.issues_detected.recent_problem_responses);
                
                document.getElementById('loading').style.display = 'none';
                document.getElementById('dashboard').style.display = 'block';
                
            } catch (error) {
                document.getElementById('loading').innerHTML = '<h3 style="color: red;">Error loading data: ' + error.message + '</h3>';
            }
        }

        function createQualityChart(data) {
            const ctx = document.getElementById('qualityChart').getContext('2d');
            
            if (qualityChart) {
                qualityChart.destroy();
            }
            
            qualityChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: data.map((_, i) => 'Response ' + (i + 1)),
                    datasets: [{
                        label: 'Overall Quality',
                        data: data.map(d => d.overall_quality),
                        borderColor: '#667eea',
                        backgroundColor: 'rgba(102, 126, 234, 0.1)',
                        tension: 0.4
                    }, {
                        label: 'Hallucination Score',
                        data: data.map(d => d.hallucination_score),
                        borderColor: '#27ae60',
                        backgroundColor: 'rgba(39, 174, 96, 0.1)',
                        tension: 0.4
                    }, {
                        label: 'Accuracy Score',
                        data: data.map(d => d.accuracy_score),
                        borderColor: '#2980b9',
                        backgroundColor: 'rgba(41, 128, 185, 0.1)',
                        tension: 0.4
                    }]
                },
                options: {
                    responsive: true,
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: 100,
                            title: {
                                display: true,
                                text: 'Score (%)'
                            }
                        }
                    },
                    plugins: {
                        legend: {
                            position: 'top',
                        }
                    }
                }
            });
        }

        function updateIssuesList(issues) {
            const issuesList = document.getElementById('issuesList');
            if (issues.length === 0) {
                issuesList.innerHTML = '<div class="issue-item" style="background: #d4edda; border-color: #c3e6cb;">✅ No critical issues detected recently!</div>';
                return;
            }
            
            issuesList.innerHTML = issues.map(issue => 
                `<div class="issue-item">
                    <strong>${issue.model.toUpperCase()}</strong> - Quality: ${issue.quality_score}% - ${issue.main_issue}
                    <br><small>${new Date(issue.timestamp).toLocaleString()}</small>
                </div>`
            ).join('');
        }

        // Auto-refresh every 30 seconds
        setInterval(loadDashboard, 30000);
        
        // Initial load
        loadDashboard();
    </script>
</body>
</html>
    """
    return html_content

@app.route("/api_stats", methods=["GET"])
def get_api_statistics():
    """Comprehensive API statistics and analytics endpoint"""
    try:
        # Calculate real-time accuracy metrics
        accuracy_metrics = calculate_accuracy_metrics()
        
        # Get recent activity (last 24 hours)
        current_time = datetime.datetime.now()
        last_24h = current_time - datetime.timedelta(hours=24)
        
        recent_calls = [
            call for call in api_call_stats["detailed_calls"]
            if datetime.datetime.fromisoformat(call["timestamp"]) > last_24h
        ]
        
        # Calculate hourly distribution for last 24 hours
        hourly_distribution = {}
        for i in range(24):
            hour_time = current_time - datetime.timedelta(hours=i)
            hour_key = hour_time.strftime("%Y-%m-%d-%H")
            hourly_distribution[hour_time.strftime("%H:00")] = api_call_stats["calls_by_hour"].get(hour_key, 0)
        
        # Get top performing endpoints
        endpoint_performance = dict(api_call_stats["calls_by_endpoint"])
        
        # Get recent errors
        recent_errors = [
            {
                "timestamp": call["timestamp"],
                "endpoint": call["endpoint"],
                "error": call["error_details"],
                "execution_time": call["execution_time"]
            }
            for call in recent_calls
            if not call["success"] and call["error_details"]
        ][-10:]  # Last 10 errors
        
        # Calculate performance percentiles
        response_times = api_call_stats["response_times"]
        performance_percentiles = {}
        if response_times:
            performance_percentiles = {
                "p50": round(np.percentile(response_times, 50) * 1000, 2),
                "p90": round(np.percentile(response_times, 90) * 1000, 2),
                "p95": round(np.percentile(response_times, 95) * 1000, 2),
                "p99": round(np.percentile(response_times, 99) * 1000, 2)
            }
        
        # Get dataset coverage statistics
        total_exercises_in_csv = len(df)
        unique_exercises_accessed = len(set([
            call["search_metrics"].get("exercise_name", "")
            for call in api_call_stats["detailed_calls"]
            if call.get("search_metrics") and call["search_metrics"].get("exercise_name")
        ]))
        
        dataset_coverage = {
            "total_exercises_in_dataset": total_exercises_in_csv,
            "unique_exercises_accessed": unique_exercises_accessed,
            "coverage_percentage": round((unique_exercises_accessed / total_exercises_in_csv * 100), 2) if total_exercises_in_csv > 0 else 0
         }
         
        response_data = {
             "api_overview": {
                 "total_api_calls": api_call_stats["total_calls"],
                 "successful_calls": api_call_stats["successful_calls"],
                 "failed_calls": api_call_stats["failed_calls"],
                 "calls_last_24h": len(recent_calls),
                 "uptime_start": "Server restart needed to track uptime",
                 "last_updated": current_time.isoformat()
             },
             "accuracy_and_performance": accuracy_metrics,
             "dataset_utilization": {
                 "total_exercises_in_dataset": dataset_coverage["total_exercises_in_dataset"],
                 "unique_exercises_accessed": dataset_coverage["unique_exercises_accessed"],
                 "coverage_percentage": dataset_coverage["coverage_percentage"],
                "csv_direct_access": api_call_stats["model_context_usage"]["csv_data_used"],
                "rag_semantic_search": api_call_stats["model_context_usage"]["rag_retrievals"],
                "empty_or_failed_retrievals": api_call_stats["model_context_usage"]["empty_responses"]
            },
            "semantic_search_analysis": {
                "total_searches": api_call_stats["semantic_search_stats"]["total_searches"],
                "average_similarity_scores": {
                    "mean": round(np.mean(api_call_stats["semantic_search_stats"]["similarity_scores"]), 4) if api_call_stats["semantic_search_stats"]["similarity_scores"] else 0,
                    "std": round(np.std(api_call_stats["semantic_search_stats"]["similarity_scores"]), 4) if api_call_stats["semantic_search_stats"]["similarity_scores"] else 0,
                    "min": round(np.min(api_call_stats["semantic_search_stats"]["similarity_scores"]), 4) if api_call_stats["semantic_search_stats"]["similarity_scores"] else 0,
                    "max": round(np.max(api_call_stats["semantic_search_stats"]["similarity_scores"]), 4) if api_call_stats["semantic_search_stats"]["similarity_scores"] else 0
                },
                "search_effectiveness": {
                    "week_matching_accuracy": round((api_call_stats["semantic_search_stats"]["week_match_accuracy"] / api_call_stats["semantic_search_stats"]["total_searches"] * 100), 2) if api_call_stats["semantic_search_stats"]["total_searches"] > 0 else 0,
                    "time_matching_accuracy": round((api_call_stats["semantic_search_stats"]["time_match_accuracy"] / api_call_stats["semantic_search_stats"]["total_searches"] * 100), 2) if api_call_stats["semantic_search_stats"]["total_searches"] > 0 else 0,
                    "exercise_name_accuracy": round((api_call_stats["semantic_search_stats"]["exercise_name_accuracy"] / api_call_stats["semantic_search_stats"]["total_searches"] * 100), 2) if api_call_stats["semantic_search_stats"]["total_searches"] > 0 else 0
                }
            },
            "model_performance": {
                "gemini_vs_gemma": {
                    "gemini_calls": api_call_stats["model_context_usage"]["gemini_calls"],
                    "gemma_calls": api_call_stats["model_context_usage"]["gemma_calls"],
                    "gemini_percentage": round((api_call_stats["model_context_usage"]["gemini_calls"] / api_call_stats["total_calls"] * 100), 2) if api_call_stats["total_calls"] > 0 else 0,
                    "gemma_percentage": round((api_call_stats["model_context_usage"]["gemma_calls"] / api_call_stats["total_calls"] * 100), 2) if api_call_stats["total_calls"] > 0 else 0
                },
                "response_time_analysis": {
                    "average_ms": round(np.mean(response_times) * 1000, 2) if response_times else 0,
                    "percentiles_ms": performance_percentiles,
                    "fastest_response_ms": round(np.min(response_times) * 1000, 2) if response_times else 0,
                    "slowest_response_ms": round(np.max(response_times) * 1000, 2) if response_times else 0
                }
            },
            "endpoint_usage": {
                "calls_by_endpoint": dict(endpoint_performance),
                "hourly_distribution_last_24h": hourly_distribution
            },
            "error_analysis": {
                "total_errors": api_call_stats["failed_calls"],
                "error_rate_percentage": round((api_call_stats["failed_calls"] / api_call_stats["total_calls"] * 100), 2) if api_call_stats["total_calls"] > 0 else 0,
                "recent_errors": recent_errors
            },
            "technical_details": {
                "vector_store_status": "Active" if vectorstore else "Not initialized",
                "csv_dataset_rows": len(df),
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                "similarity_search_k_value": 4,
                "storage_limit": "Last 1000 detailed calls kept in memory"
            }
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({
            "error": "Failed to generate API statistics",
            "details": str(e),
            "basic_stats": {
                "total_calls": api_call_stats["total_calls"],
                "successful_calls": api_call_stats["successful_calls"],
                "failed_calls": api_call_stats["failed_calls"]
            }
        }), 500


@app.route("/feedback", methods=["POST"])
def get_feedback():
    data = request.get_json()
    week_pregnancy = data.get("week_pregnancy")
    n_sets = data.get("n_sets")
    time_val = data.get("time")
    name = data.get("name", "")

    # Ensure time_param alias for backward compatibility with prompt templates
    time_param = time_val

    if not all([week_pregnancy, n_sets, time_param]):
        return jsonify({"error": "Missing parameters"}), 400

    try:
        week_pregnancy = int(week_pregnancy)
        n_sets = int(n_sets)
    except ValueError:
        return jsonify({"error": "week_pregnancy and n_sets must be integers"}), 400

    # Enhanced data retrieval with ALL dataset columns
    exact_matches = []
    if name:
        name_lower = name.lower().strip()
        time_lower = time_param.lower().strip()

        # Filter dataframe for exact matches
        matches = df[(df['name'].str.lower() == name_lower)
                     & (df['week'] == week_pregnancy)]

        if not matches.empty:
            # If we have time match, prioritize it
            time_matches = matches[matches['time of day'].str.lower(
            ) == time_lower]
            filtered_matches = time_matches if not time_matches.empty else matches

            # Convert to our enhanced metadata format
            for _, row in filtered_matches.iterrows():
                exact_matches.append({
                    "id": row.get('id'),
                    "name": row.get('name'),
                    "week": row.get('week'),
                    "time": row.get('time of day'),
                    "N": row.get('N'),
                    "benefits": row.get('benefits'),
                    "link": row.get('Link'),
                    "trimester": row.get('trimester'),
                    "contraindications": row.get('contraindications'),
                    "modifications": row.get('modifications'),
                    "intensity": row.get('intensity'),
                    "rest_interval": row.get('rest_interval'),
                    "equipment_needed": row.get('equipment_needed'),
                    "primary_muscles": row.get('primary_muscles'),
                    "safety_tips": row.get('safety_tips'),
                    "fitness_level": row.get('fitness_level'),
                    "medical_clearance": row.get('medical_clearance'),
                    "progression_guidelines": row.get('progression_guidelines'),
                    "postpartum_relevance": row.get('postpartum_relevance')
                })

    # If we found exact matches, use them exclusively
    if exact_matches:
        relevant_exercises = exact_matches
    else:
        # If no exact matches or no name provided, use RAG with enhanced query
        if name:
            query = f"Exercise name: {name}, Pregnancy week: {week_pregnancy}, Time of day: {time_param}"
        else:
            query = f"Exercises for pregnancy week: {week_pregnancy}, Time of day: {time_param}"

        # Retrieve relevant exercises using RAG
        docs = vectorstore.similarity_search(query, k=4)

        # Extract exercises from metadata with enhanced data
        relevant_exercises = []
        for doc in docs:
            metadata = doc.metadata

            # Apply post-retrieval filtering for better match quality
            # Prioritize exact week matches
            if 'week' in metadata and metadata['week'] == week_pregnancy:
                # Get full row data from dataframe using row_index
                row_index = metadata.get('row_index')
                if row_index is not None and row_index < len(df):
                    row = df.iloc[row_index]
                    exercise_info = {
                        "id": row.get('id'),
                        "name": row.get('name'),
                        "week": row.get('week'),
                        "time": row.get('time of day'),
                        "N": row.get('N'),
                        "benefits": row.get('benefits'),
                        "link": row.get('Link'),
                        "trimester": row.get('trimester'),
                        "contraindications": row.get('contraindications'),
                        "modifications": row.get('modifications'),
                        "intensity": row.get('intensity'),
                        "rest_interval": row.get('rest_interval'),
                        "equipment_needed": row.get('equipment_needed'),
                        "primary_muscles": row.get('primary_muscles'),
                        "safety_tips": row.get('safety_tips'),
                        "fitness_level": row.get('fitness_level'),
                        "medical_clearance": row.get('medical_clearance'),
                        "progression_guidelines": row.get('progression_guidelines'),
                        "postpartum_relevance": row.get('postpartum_relevance')
                    }
                    relevant_exercises.append(exercise_info)

        # If we didn't find any exercises after filtering by week, use the top results regardless
        if not relevant_exercises and docs:
            for doc in docs:
                metadata = doc.metadata
                row_index = metadata.get('row_index')
                if row_index is not None and row_index < len(df):
                    row = df.iloc[row_index]
                    exercise_info = {
                        "id": row.get('id'),
                        "name": row.get('name'),
                        "week": row.get('week'),
                        "time": row.get('time of day'),
                        "N": row.get('N'),
                        "benefits": row.get('benefits'),
                        "link": row.get('Link'),
                        "trimester": row.get('trimester'),
                        "contraindications": row.get('contraindications'),
                        "modifications": row.get('modifications'),
                        "intensity": row.get('intensity'),
                        "rest_interval": row.get('rest_interval'),
                        "equipment_needed": row.get('equipment_needed'),
                        "primary_muscles": row.get('primary_muscles'),
                        "safety_tips": row.get('safety_tips'),
                        "fitness_level": row.get('fitness_level'),
                        "medical_clearance": row.get('medical_clearance'),
                        "progression_guidelines": row.get('progression_guidelines'),
                        "postpartum_relevance": row.get('postpartum_relevance')
                    }
                    relevant_exercises.append(exercise_info)

    # Enhanced prompt for structured analysis using Gemma
    exercise_data = relevant_exercises[0] if relevant_exercises else {}

    prompt = f"""As a certified prenatal fitness expert, analyze this exercise session and provide comprehensive feedback in JSON format.

USER SESSION DATA:
- Exercise Name: {name if name else 'Not specified'}
- Pregnancy Week: {week_pregnancy}
- Sets Performed: {n_sets}
- Time of Day: {time_param}

REFERENCE EXERCISE DATA:
{exercise_data}

Provide structured analysis including:
1. Exercise analysis (name, recommended vs performed sets, timing, week suitability)
2. Technical details (muscles, intensity, equipment, rest intervals, fitness level, trimester)
3. Safety guidelines (contraindications, safety tips, medical clearance, modifications)
4. Benefits and progression (benefits, progression guidelines, postpartum relevance)
5. Recommendations (continue routine, modifications, next week adjustments, warning signs)
6. Summary (overall assessment, key points, reference link)

Generate a detailed JSON response with all categories filled. If data is missing, use null values."""

    # Generate feedback with Gemma using enhanced prompt
    generated_output = gemma(prompt, max_new_tokens=1200,
                             truncation=True, return_full_text=False)

    response_content = ""
    if generated_output and isinstance(generated_output, list) and generated_output[0].get("generated_text"):
        response_content = generated_output[0]["generated_text"].strip()

        # Try to extract JSON from the response
        json_start = response_content.find('{')
        json_end = response_content.rfind('}') + 1

        if json_start != -1 and json_end > json_start:
            json_str = response_content[json_start:json_end]
            try:
                parsed_response = json.loads(json_str)
                return jsonify(parsed_response)
            except json.JSONDecodeError:
                # If JSON parsing fails, return the raw response as before
                return jsonify({"feedback": response_content})
        else:
            # If no JSON structure found, return raw response
            return jsonify({"feedback": response_content})

    return jsonify({"feedback": "Unable to generate feedback at this time."})

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
            retrieved_docs = app.food_vectorstore.similarity_search(
                query, k=5)  # Increased k

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
                        print(
                            f"Exact match found for '{food_name}': {doc.metadata.get('name')}")
                        break

                # If no exact match found, fall back to the top semantic match
                if not exact_match_found:
                    best_match = retrieved_docs[0].metadata
                    print(
                        f"No exact match for '{food_name}', using top semantic match: {best_match.get('name')}")

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
            elif review_response:  # Log if response format is unexpected but not an exception
                print(
                    f"Gemma review generation returned unexpected format: {review_response}")
            else:  # Log if response is None or empty
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
            gemini_1_5_flash_model = genai.GenerativeModel(
                'gemini-1.5-flash')  # Explicitly use gemini-1.5-flash
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

            response = gemini_1_5_flash_model.generate_content(  # Use the specific gemini-1.5-flash model instance
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
                        print(
                            f"Exact match found for '{food_name}': {doc.metadata.get('name')}")
                        break
                if not exact_match_found:
                    best_match = retrieved_docs[0].metadata
                    print(
                        f"No exact match for '{food_name}', using top semantic match: {best_match.get('name')}")

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
                            total_calories += float(
                                nutrition_data.get("calories", 100))
                            total_protein += float(nutrition_data.get("protein", 2))
                        except json.JSONDecodeError:
                            food_info.update(
                                {"calories": 100, "protein": 2, "estimated": True, "error": "Gemma estimation JSON parse failed"})
                            total_calories += 100
                            total_protein += 2
                    else:
                        food_info.update(
                            {"calories": 100, "protein": 2, "estimated": True, "error": "Gemma estimation no JSON found"})
                        total_calories += 100
                        total_protein += 2
                else:
                    food_info.update(
                        {"calories": 100, "protein": 2, "estimated": True, "error": "Gemma estimation failed"})
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
            elif review_response:  # Log if response format is unexpected but not an exception
                print(
                    f"Gemma review generation returned unexpected format: {review_response}")
            else:  # Log if response is None or empty
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
                app.timeline_df = pd.read_csv(
                    "timeline_dataset.csv", encoding='utf-8')
            except UnicodeDecodeError:
                try:
                    # Try Latin-1 encoding which is more permissive
                    app.timeline_df = pd.read_csv(
                        "timeline_dataset.csv", encoding='latin1')
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
                    app.embeddings = HuggingFaceEmbeddings(
                        model_name="sentence-transformers/all-MiniLM-L6-v2")

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
    data = request.get_json(silent=True)  # Expects JSON

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


@app.route("/feedback_gemini", methods=["POST"])
def get_feedback_gemini():
    if not gemini_model_client:
        return jsonify({"error": "Gemini client is not configured."}), 503
    
    data = request.get_json()
    week_pregnancy = data.get("week_pregnancy")
    n_sets = data.get("n_sets")
    time_val = data.get("time")
    name = data.get("name", "")

    # Ensure time_param alias for backward compatibility with prompt templates
    time_param = time_val

    if not all([week_pregnancy, n_sets, time_param]):
        return jsonify({"error": "Missing parameters"}), 400

    try:
        week_pregnancy = int(week_pregnancy)
        n_sets = int(n_sets)
    except ValueError:
        return jsonify({"error": "week_pregnancy and n_sets must be integers"}), 400

    # Enhanced data retrieval with ALL dataset columns
    exact_matches = []
    if name:
        name_lower = name.lower().strip()
        time_lower = time_param.lower().strip()
        matches = df[(df['name'].str.lower() == name_lower)
                     & (df['week'] == week_pregnancy)]

        if not matches.empty:
            time_matches = matches[matches['time of day'].str.lower(
            ) == time_lower]
            filtered_matches = time_matches if not time_matches.empty else matches

            for _, row in filtered_matches.iterrows():
                # Extract ALL available data from the dataset
                exact_matches.append({
                    "id": row.get('id'),
                    "name": row.get('name'),
                    "week": row.get('week'),
                    "time": row.get('time of day'),
                    "N": row.get('N'),
                    "benefits": row.get('benefits'),
                    "link": row.get('Link'),
                    "trimester": row.get('trimester'),
                    "contraindications": row.get('contraindications'),
                    "modifications": row.get('modifications'),
                    "intensity": row.get('intensity'),
                    "rest_interval": row.get('rest_interval'),
                    "equipment_needed": row.get('equipment_needed'),
                    "primary_muscles": row.get('primary_muscles'),
                    "safety_tips": row.get('safety_tips'),
                    "fitness_level": row.get('fitness_level'),
                    "medical_clearance": row.get('medical_clearance'),
                    "progression_guidelines": row.get('progression_guidelines'),
                    "postpartum_relevance": row.get('postpartum_relevance')
                })

    if exact_matches:
        relevant_exercises = exact_matches
    else:
        # RAG fallback with enhanced metadata extraction
        if name:
            query = f"Exercise name: {name}, Pregnancy week: {week_pregnancy}, Time of day: {time_param}"
        else:
            query = f"Exercises for pregnancy week: {week_pregnancy}, Time of day: {time_param}"

        docs = vectorstore.similarity_search(query, k=4)
        relevant_exercises = []

        for doc in docs:
            metadata = doc.metadata
            if 'week' in metadata and metadata['week'] == week_pregnancy:
                # Get full row data from dataframe using row_index
                row_index = metadata.get('row_index')
                if row_index is not None and row_index < len(df):
                    row = df.iloc[row_index]
                    exercise_info = {
                        "id": row.get('id'),
                        "name": row.get('name'),
                        "week": row.get('week'),
                        "time": row.get('time of day'),
                        "N": row.get('N'),
                        "benefits": row.get('benefits'),
                        "link": row.get('Link'),
                        "trimester": row.get('trimester'),
                        "contraindications": row.get('contraindications'),
                        "modifications": row.get('modifications'),
                        "intensity": row.get('intensity'),
                        "rest_interval": row.get('rest_interval'),
                        "equipment_needed": row.get('equipment_needed'),
                        "primary_muscles": row.get('primary_muscles'),
                        "safety_tips": row.get('safety_tips'),
                        "fitness_level": row.get('fitness_level'),
                        "medical_clearance": row.get('medical_clearance'),
                        "progression_guidelines": row.get('progression_guidelines'),
                        "postpartum_relevance": row.get('postpartum_relevance')
                    }
                    relevant_exercises.append(exercise_info)

        # If no exact week matches, include broader results
        if not relevant_exercises and docs:
            for doc in docs:
                metadata = doc.metadata
                row_index = metadata.get('row_index')
                if row_index is not None and row_index < len(df):
                    row = df.iloc[row_index]
                    exercise_info = {
                        "id": row.get('id'),
                        "name": row.get('name'),
                        "week": row.get('week'),
                        "time": row.get('time of day'),
                        "N": row.get('N'),
                        "benefits": row.get('benefits'),
                        "link": row.get('Link'),
                        "trimester": row.get('trimester'),
                        "contraindications": row.get('contraindications'),
                        "modifications": row.get('modifications'),
                        "intensity": row.get('intensity'),
                        "rest_interval": row.get('rest_interval'),
                        "equipment_needed": row.get('equipment_needed'),
                        "primary_muscles": row.get('primary_muscles'),
                        "safety_tips": row.get('safety_tips'),
                        "fitness_level": row.get('fitness_level'),
                        "medical_clearance": row.get('medical_clearance'),
                        "progression_guidelines": row.get('progression_guidelines'),
                        "postpartum_relevance": row.get('postpartum_relevance')
                    }
                    relevant_exercises.append(exercise_info)

    # Enhanced prompt for structured analysis
    exercise_data = relevant_exercises[0] if relevant_exercises else {}

    prompt = f"""As a certified prenatal fitness expert, analyze this exercise session and provide comprehensive feedback in the exact JSON structure below.

USER SESSION DATA:
- Exercise Name: {name if name else 'Not specified'}
- Pregnancy Week: {week_pregnancy}
- Sets Performed: {n_sets}
- Time of Day: {time_param}

REFERENCE EXERCISE DATA:
{exercise_data}

Provide your analysis in this EXACT JSON structure (include ALL fields even if null):

{{
  "exercise_analysis": {{
         "exercise_name": "{exercise_data.get('name', name) if exercise_data.get('name') or name else 'null'}",
     "recommended_sets": "{exercise_data.get('N') if exercise_data.get('N') else 'null'}",
    "user_performed_sets": {n_sets},
    "sets_comparison": "analysis of user sets vs recommended",
    "timing_appropriateness": "analysis of time of day choice",
    "week_suitability": "analysis for current pregnancy week"
  }},
  "technical_details": {{
         "primary_muscles": "{exercise_data.get('primary_muscles') if exercise_data.get('primary_muscles') else 'null'}",
     "intensity_level": "{exercise_data.get('intensity') if exercise_data.get('intensity') else 'null'}",
     "equipment_needed": "{exercise_data.get('equipment_needed') if exercise_data.get('equipment_needed') else 'null'}",
     "rest_interval": "{exercise_data.get('rest_interval') if exercise_data.get('rest_interval') else 'null'}",
     "fitness_level_required": "{exercise_data.get('fitness_level') if exercise_data.get('fitness_level') else 'null'}",
     "trimester": "{exercise_data.get('trimester') if exercise_data.get('trimester') else 'null'}"
  }},
  "safety_guidelines": {{
         "contraindications": "{exercise_data.get('contraindications') if exercise_data.get('contraindications') else 'null'}",
     "safety_tips": "{exercise_data.get('safety_tips') if exercise_data.get('safety_tips') else 'null'}",
     "medical_clearance_required": "{exercise_data.get('medical_clearance') if exercise_data.get('medical_clearance') else 'null'}",
     "modifications_available": "{exercise_data.get('modifications') if exercise_data.get('modifications') else 'null'}"
  }},
  "benefits_and_progression": {{
         "exercise_benefits": "{exercise_data.get('benefits') if exercise_data.get('benefits') else 'null'}",
     "progression_guidelines": "{exercise_data.get('progression_guidelines') if exercise_data.get('progression_guidelines') else 'null'}",
     "postpartum_relevance": "{exercise_data.get('postpartum_relevance') if exercise_data.get('postpartum_relevance') else 'null'}"
  }},
  "recommendations": {{
    "continue_current_routine": "boolean and reasoning",
    "suggested_modifications": "specific modifications if needed",
    "next_week_adjustments": "recommendations for week {week_pregnancy + 1}",
    "warning_signs_to_watch": "specific warning signs for this exercise and week"
  }},
  "summary": {{
    "overall_assessment": "brief overall assessment",
    "key_points": ["point 1", "point 2", "point 3"],
         "reference_link": "{exercise_data.get('link') if exercise_data.get('link') else 'null'}"
  }}
}}

IMPORTANT: 
- Replace null with actual null values in JSON, not the string "null"
- Provide specific, evidence-based analysis
- Keep responses concise but comprehensive
- If no data available for a field, use null
- Ensure valid JSON format"""

    try:
        response = gemini_model_client.generate_content(prompt)
        response_content = response.text.strip() if response else ""

        # Extract and parse JSON response
        json_start = response_content.find('{')
        json_end = response_content.rfind('}') + 1

        if json_start != -1 and json_end > json_start:
            json_str = response_content[json_start:json_end]
            try:
                parsed_response = json.loads(json_str)
                
                # AI Response Quality Analysis (for internal tracking only)
                user_data = {
                    "name": name,
                    "week_pregnancy": week_pregnancy,
                    "n_sets": n_sets,
                    "time": time_param
                }
                csv_data_used = relevant_exercises[0] if relevant_exercises else None
                analysis = analyze_ai_response_quality(user_data, parsed_response, csv_data_used, "gemini", "/feedback_gemini")
                
                return jsonify(parsed_response)
            except json.JSONDecodeError as e:
                return jsonify({
                    "error": "Failed to parse structured response",
                    "details": str(e),
                    "raw_response": response_content
                }), 500
        else:
            # AI Analytics for non-JSON response (internal tracking only)
            user_data = {
                "name": name,
                "week_pregnancy": week_pregnancy,
                "n_sets": n_sets,
                "time": time_param
            }
            csv_data_used = relevant_exercises[0] if relevant_exercises else None
            analysis = analyze_ai_response_quality(user_data, response_content, csv_data_used, "gemini", "/feedback_gemini")
            
            return jsonify({
                "error": "No valid JSON structure found in response",
                "raw_response": response_content
            }), 500

    except Exception as e:
        return jsonify({
            "error": "Error processing request with Gemini model",
            "details": str(e)
        }), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
