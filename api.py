from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd
import os
from flask_cors import CORS
import io
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
import google.generativeai as genai
from werkzeug.utils import secure_filename
import json
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                           confusion_matrix, classification_report, roc_auc_score)
import xgboost as xgb
from catboost import CatBoostClassifier
import warnings

warnings.filterwarnings('ignore')

# Configure Gemini
GEMINI_MODEL_NAME = "gemini-2.0-flash"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# Upload configuration
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tiff', 'bmp', 'webp', 'pdf'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
CORS(app)

# Configure Tesseract (adjust path if needed)
try:
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
except:
    pass  # Use system PATH

# Initialize Gemini
try:
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model_client = genai.GenerativeModel(GEMINI_MODEL_NAME)
        print("Gemini client configured successfully")
    else:
        gemini_model_client = None
        print("GEMINI_API_KEY not found. Image processing will be limited.")
except Exception as e:
    gemini_model_client = None
    print(f"Error configuring Gemini: {e}")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_medical_image(image_file, prediction_type="preterm"):
    """Process medical image and extract structured data using Gemini"""
    if not gemini_model_client:
        return {"error": "Gemini not configured. Cannot process images."}, 400
    
    try:
        # Convert image for Gemini
        pil_image = Image.open(image_file.stream)
        if pil_image.mode not in ['RGB', 'RGBA']:
            pil_image = pil_image.convert('RGB')
        
        img_byte_arr = io.BytesIO()
        pil_image.save(img_byte_arr, format='JPEG')
        image_bytes = img_byte_arr.getvalue()
        image_part = {"mime_type": "image/jpeg", "data": image_bytes}
        
        # Also try OCR as backup
        ocr_text = ""
        try:
            # Enhance image for OCR
            img_enhanced = pil_image.convert('L')
            enhancer_contrast = ImageEnhance.Contrast(img_enhanced)
            img_enhanced = enhancer_contrast.enhance(1.5)
            img_enhanced = img_enhanced.filter(ImageFilter.SHARPEN)
            ocr_text = pytesseract.image_to_string(img_enhanced)
        except Exception as ocr_e:
            print(f"OCR failed: {ocr_e}")
        
        # Create appropriate prompt based on prediction type
        if prediction_type == "preterm":
            prompt = """
            You are a medical data extraction expert. Analyze this medical image/lab report and extract ALL relevant information for preterm birth risk assessment.
            
            Extract and return a JSON object with the following structure (use null for missing values):
            {
                "Hb [g/dl]": null,
                "WBC [G/l]": null, 
                "PLT [G/l]": null,
                "HCT [%]": null,
                "Age": null,
                "No. of pregnancy": null,
                "No. of deliveries": null,
                "Week of sample collection": null,
                "Height": null,
                "Weight": null,
                "BMI": null,
                "Education [0-primary. 1-vocational. 2-higher]": null,
                "Marital status [0-single. 1-married]": null,
                "Gestational diabetes mellitus [0-no. 1-type 1. 2-type 2]": null,
                "Gestational hypothyroidism [0-no.-1yes]": null,
                "History of preterm labour [0-no.1-yes]": null,
                "Smoking [0-no.1-yes]": null,
                "History of surgical delivery [0-no.1-yes]": null,
                "History of caesarean section [0-no.1-yes]": null,
                "CRP": null,
                "Week of delivery": null,
                "Type of delivery [0-vaginal.1-c-section]": null
            }
            
            Important notes:
            - Extract exact numerical values where visible
            - For categorical fields, use the specified coding (0/1/2)
            - If BMI not shown but height/weight available, calculate it
            - Look for lab values, patient demographics, medical history
            - Be precise with units (g/dl, G/l, etc.)
            """
        else:  # maternal risk
            prompt = """
            You are a medical data extraction expert. Analyze this medical image/lab report and extract ALL relevant information for maternal risk assessment.
            
            Extract and return a JSON object with maternal health indicators. Look for:
            - Age, systolic/diastolic blood pressure
            - Blood sugar levels, body temperature
            - Heart rate and other vital signs
            - Any pregnancy-related measurements
            
            Return the data in a structured JSON format with appropriate field names and numerical values.
            As:
            {
                "Age": int,
                "SystolicBP": int,
                "DiastolicBP": int,
                "BS": float,
                "BodyTemp": float,
                "HeartRate": int
            }
            """
        
        # Add OCR text to prompt if available
        if ocr_text.strip():
            prompt += f"\n\nOCR extracted text from image:\n{ocr_text}\n\nUse this text along with visual analysis to extract the medical data."
        
        # Generate response
        response = gemini_model_client.generate_content([prompt, image_part])
        gemini_text = response.text.strip()
        
        # Extract JSON from response
        json_start = gemini_text.find('{')
        json_end = gemini_text.rfind('}') + 1
        
        if json_start != -1 and json_end > json_start:
            json_str = gemini_text[json_start:json_end]
            try:
                parsed_data = json.loads(json_str)
                return parsed_data, 200
            except json.JSONDecodeError as e:
                return {
                    "error": "Failed to parse medical data from image",
                    "details": str(e),
                    "raw_response": gemini_text,
                    "ocr_text": ocr_text
                }, 400
        else:
            return {
                "error": "No structured data found in image",
                "raw_response": gemini_text,
                "ocr_text": ocr_text
            }, 400
            
    except Exception as e:
        return {
            "error": f"Error processing medical image: {str(e)}",
            "details": str(e)
        }, 500

class PretermBirthPredictor:
    def __init__(self, data_path='lab_results.csv'):
        self.data_path = data_path
        self.models = {}
        self.scaler = StandardScaler()
        self.best_model = None
        self.feature_names = None
        self.results = {}
        
    def load_and_preprocess_data(self):
        print("Loading and preprocessing preterm birth data...")
        
        df = pd.read_csv(self.data_path, sep=';')
        print(f"Dataset shape: {df.shape}")
        
        df.columns = df.columns.str.strip()
        
        numeric_columns = ['Hb [g/dl]', 'WBC [G/l]', 'PLT [G/l]', 'HCT [%]', 'Age', 
                          'No. of pregnancy', 'No. of deliveries', 'Week of sample collection',
                          'Height', 'Weight', 'BMI', 'CRP', 'Week of delivery']
        
        categorical_columns = ['Education [0-primary. 1-vocational. 2-higher]',
                             'Marital status [0-single. 1-married]',
                             'Gestational diabetes mellitus [0-no. 1-type 1. 2-type 2]',
                             'Gestational hypothyroidism [0-no.-1yes]',
                             'History of preterm labour [0-no.1-yes]',
                             'Smoking [0-no.1-yes]',
                             'History of surgical delivery [0-no.1-yes]',
                             'History of caesarean section [0-no.1-yes]',
                             'Type of delivery [0-vaginal.1-c-section]']
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        for col in categorical_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
        
        df = df.dropna(subset=['Label [0-term. 1-preterm]'])
        
        for col in df.select_dtypes(include=[np.number]).columns:
            if col != 'Label [0-term. 1-preterm]':
                df[col] = df[col].fillna(df[col].median())
        
        print(f"After preprocessing: {df.shape}")
        return df
    
    def feature_engineering(self, df):
        print("Performing feature engineering...")
        
        df['BMI_category'] = pd.cut(df['BMI'], bins=[0, 18.5, 25, 30, float('inf')], 
                                   labels=[0, 1, 2, 3], include_lowest=True).astype(int)
        
        df['Age_category'] = pd.cut(df['Age'], bins=[0, 25, 35, float('inf')], 
                                   labels=[0, 1, 2], include_lowest=True).astype(int)
        
        if 'Hb [g/dl]' in df.columns:
            df['Anemia'] = (df['Hb [g/dl]'] < 11).astype(int)
        
        df['High_risk_pregnancy'] = ((df['Age'] > 35) | (df['Age'] < 18) | 
                                    (df['No. of pregnancy'] > 4) | 
                                    (df['Gestational diabetes mellitus [0-no. 1-type 1. 2-type 2]'] > 0)).astype(int)
        
        df['Gestational_age_at_sample'] = df['Week of sample collection']
        df['Early_sample'] = (df['Week of sample collection'] < 32).astype(int)
        
        return df
    
    def prepare_features(self, df):
        target_col = 'Label [0-term. 1-preterm]'
        feature_cols = [col for col in df.columns if col not in [target_col, 'sample no.']]
        
        X = df[feature_cols]
        y = df[target_col]
        
        self.feature_names = list(X.columns)
        return X, y
    
    def evaluate_model(self, model, X_train, X_test, y_train, y_test, model_name):
        print(f"Evaluating {model_name}...")
        
        model.fit(X_train, y_train)
        
        y_pred_test = model.predict(X_test)
        
        test_acc = accuracy_score(y_test, y_pred_test)
        precision = precision_score(y_test, y_pred_test)
        recall = recall_score(y_test, y_pred_test)
        f1 = f1_score(y_test, y_pred_test)
        
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
        
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)[:, 1]
            auc_score = roc_auc_score(y_test, y_proba)
        else:
            auc_score = None
        
        print(f"{model_name} - Accuracy: {test_acc:.4f}, F1: {f1:.4f}")
        
        return {
            'model': model,
            'test_acc': test_acc,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'auc_score': auc_score
        }
    
    def train_and_save_models(self):
        print("Training preterm birth prediction models...")
        
        df = self.load_and_preprocess_data()
        df = self.feature_engineering(df)
        X, y = self.prepare_features(df)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        models_config = {
            'Logistic Regression': {
                'model': LogisticRegression(random_state=42, class_weight='balanced'),
                'params': {'C': [0.1, 1, 10], 'solver': ['liblinear', 'lbfgs']}
            },
            'Linear SVM': {
                'model': SVC(kernel='linear', random_state=42, class_weight='balanced', probability=True),
                'params': {'C': [0.1, 1, 10]}
            },
            'RBF SVM': {
                'model': SVC(kernel='rbf', random_state=42, class_weight='balanced', probability=True),
                'params': {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']}
            },
            'Random Forest': {
                'model': RandomForestClassifier(random_state=42, class_weight='balanced'),
                'params': {'n_estimators': [100, 200], 'max_depth': [3, 5, 10]}
            },
            'XGBoost': {
                'model': xgb.XGBClassifier(
                    random_state=42, 
                    eval_metric='logloss',
                    enable_categorical=False,
                    use_label_encoder=False
                ),
                'params': {'n_estimators': [100, 200], 'max_depth': [3, 5], 'learning_rate': [0.1, 0.2]}
            },
            'CatBoost': {
                'model': CatBoostClassifier(random_state=42, verbose=False, auto_class_weights='Balanced'),
                'params': {'iterations': [100, 200], 'depth': [4, 6], 'learning_rate': [0.1, 0.2]}
            }
        }
        
        for model_name, config in models_config.items():
            print(f"Optimizing {model_name}...")
            
            grid_search = GridSearchCV(
                config['model'], 
                config['params'], 
                cv=3, 
                scoring='f1',
                n_jobs=-1
            )
            
            grid_search.fit(X_train_scaled, y_train)
            best_model = grid_search.best_estimator_
            
            result = self.evaluate_model(best_model, X_train_scaled, X_test_scaled, y_train, y_test, model_name)
            
            self.models[model_name] = best_model
            self.results[model_name] = result
        
        top_models = sorted(self.results.items(), key=lambda x: x[1]['f1_score'], reverse=True)[:3]
        ensemble_estimators = [(name, self.models[name]) for name, _ in top_models]
        ensemble_model = VotingClassifier(estimators=ensemble_estimators, voting='soft')
        
        result = self.evaluate_model(ensemble_model, X_train_scaled, X_test_scaled, y_train, y_test, "Ensemble")
        self.models['Ensemble'] = ensemble_model
        self.results['Ensemble'] = result
        
        best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['f1_score'])
        self.best_model = best_model_name
        
        print(f"Best model: {best_model_name} (F1: {self.results[best_model_name]['f1_score']:.4f})")
        
        self.save_models()
    
    def save_models(self):
        if not os.path.exists('saved_models'):
            os.makedirs('saved_models')
        
        with open('saved_models/preterm_scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        
        for model_name, model in self.models.items():
            filename = f"saved_models/preterm_{model_name.lower().replace(' ', '_')}_model.pkl"
            with open(filename, 'wb') as f:
                pickle.dump(model, f)
        
        model_info = {
            'feature_names': self.feature_names,
            'best_model': self.best_model,
            'results': {name: {k: v for k, v in result.items() if k != 'model'} 
                       for name, result in self.results.items()},
            'class_mapping': {0: 'Term Birth', 1: 'Preterm Birth'}
        }
        
        with open('saved_models/preterm_model_info.pkl', 'wb') as f:
            pickle.dump(model_info, f)
        
        print("Preterm birth models saved successfully!")
    
    def load_models(self):
        try:
            with open('saved_models/preterm_birth_scaler.pkl', 'rb') as f:
                self.scaler = pickle.load(f)
            
            with open('saved_models/preterm_birth_model_info.pkl', 'rb') as f:
                model_info = pickle.load(f)
                self.feature_names = model_info['feature_names']
                self.best_model = model_info['best_model']
                self.results = model_info['results']
            
            model_files = {
                'Logistic Regression': 'saved_models/preterm_birth_logistic_regression_model.pkl',
                'Linear SVM': 'saved_models/preterm_birth_linear_svm_model.pkl',
                'RBF SVM': 'saved_models/preterm_birth_rbf_svm_model.pkl',
                'Random Forest': 'saved_models/preterm_birth_random_forest_model.pkl',
                'XGBoost': 'saved_models/preterm_birth_xgboost_model.pkl',
                'CatBoost': 'saved_models/preterm_birth_catboost_model.pkl',
                'Ensemble': 'saved_models/preterm_birth_ensemble_model.pkl'
            }
            
            for model_name, filename in model_files.items():
                if os.path.exists(filename):
                    with open(filename, 'rb') as f:
                        self.models[model_name] = pickle.load(f)
            
            print(f"Loaded {len(self.models)} preterm birth models")
            return True
            
        except Exception as e:
            print(f"Could not load preterm birth models: {e}")
            return False
    
    def predict_preterm_risk(self, patient_data, model_name=None):
        if not self.models:
            return {"error": "No models loaded"}
        
        try:
            if isinstance(patient_data, dict):
                patient_df = pd.DataFrame([patient_data])
            else:
                patient_df = patient_data.copy()
            
            patient_df = self.feature_engineering(patient_df)
            patient_features = patient_df[self.feature_names]
            patient_scaled = self.scaler.transform(patient_features)
            
            if model_name and model_name in self.models:
                # Return prediction from specific model
                model = self.models[model_name]
                prediction = model.predict(patient_scaled)[0]
                
                if hasattr(model, 'predict_proba'):
                    probability = model.predict_proba(patient_scaled)[0]
                    risk_probability = probability[1]
                else:
                    risk_probability = None
                
                return {
                    'model': model_name,
                    'prediction': 'Preterm Birth' if prediction == 1 else 'Term Birth',
                    'risk_score': int(prediction),
                    'probability': float(risk_probability) if risk_probability is not None else None
                }
            else:
                # Return predictions from ALL models
                results = {}
                for name, model in self.models.items():
                    pred = model.predict(patient_scaled)[0]
                    prob = None
                    if hasattr(model, 'predict_proba'):
                        try:
                            proba = model.predict_proba(patient_scaled)[0]
                            prob = float(proba[1])  # Probability of preterm birth
                        except:
                            pass
                    
                    results[name] = {
                        'prediction': 'Preterm Birth' if pred == 1 else 'Term Birth',
                        'risk_score': int(pred),
                        'probability': prob
                    }
                
                return results
            
        except Exception as e:
            return {"error": f"Prediction failed: {str(e)}"}

class MaternalRiskPredictor:
    def __init__(self):
        self.models = {}
        self.scaler = None
        self.model_info = None
        self.load_models()
        
        # If models fail to load due to compatibility issues, retrain them
        if not self.models and os.path.exists('maternal_dataset.csv'):
            print("Models failed to load. Retraining with current XGBoost version...")
            self.train_and_save_models()
    
    def load_models(self):
        try:
            with open('saved_models/scaler.pkl', 'rb') as f:
                self.scaler = pickle.load(f)
            
            with open('saved_models/model_info.pkl', 'rb') as f:
                self.model_info = pickle.load(f)
            
            model_files = {
                'SVM': 'saved_models/svm_model.pkl',
                'XGBoost': 'saved_models/xgboost_model.pkl',
                'Random Forest': 'saved_models/random_forest_model.pkl',
                'Decision Tree': 'saved_models/decision_tree_model.pkl',
                'Ensemble': 'saved_models/ensemble_model.pkl'
            }
            
            for model_name, filename in model_files.items():
                if os.path.exists(filename):
                    try:
                        with open(filename, 'rb') as f:
                            model = pickle.load(f)
                            
                        # Fix XGBoost compatibility issue
                        if model_name == 'XGBoost' and hasattr(model, 'use_label_encoder'):
                            print(f"Fixing XGBoost model compatibility...")
                            # Remove the deprecated attribute
                            delattr(model, 'use_label_encoder')
                            
                        self.models[model_name] = model
                        
                    except Exception as model_error:
                        print(f"Error loading {model_name} model: {model_error}")
                        # Skip this model but continue with others
                        continue
            
            print(f"Loaded {len(self.models)} maternal risk models")
            
        except Exception as e:
            print(f"Error loading maternal risk models: {e}")
            self.models = {}
    
    def predict_single(self, data, model_name=None):
        if not self.models:
            return {"error": "No maternal risk models loaded"}
        
        try:
            features = np.array([data]).reshape(1, -1)
            features_scaled = self.scaler.transform(features)
            
            if model_name and model_name in self.models:
                prediction = self.models[model_name].predict(features_scaled)[0]
                probability = None
                if hasattr(self.models[model_name], 'predict_proba'):
                    try:
                        proba = self.models[model_name].predict_proba(features_scaled)[0]
                        probability = {
                            self.model_info['class_mapping'][i]: round(float(prob), 4) 
                            for i, prob in enumerate(proba)
                        }
                    except:
                        pass
                
                return {
                    "model": model_name,
                    "prediction": self.model_info['class_mapping'][int(prediction)],
                    "prediction_code": int(prediction),
                    "probability": probability
                }
            else:
                results = {}
                for name, model in self.models.items():
                    pred = model.predict(features_scaled)[0]
                    prob = None
                    if hasattr(model, 'predict_proba'):
                        try:
                            proba = model.predict_proba(features_scaled)[0]
                            prob = {
                                self.model_info['class_mapping'][i]: round(float(p), 4) 
                                for i, p in enumerate(proba)
                            }
                        except:
                            pass
                    
                    results[name] = {
                        "prediction": self.model_info['class_mapping'][int(pred)],
                        "prediction_code": int(pred),
                        "probability": prob
                    }
                
                return results
                
        except Exception as e:
            return {"error": f"Prediction failed: {str(e)}"}
    
    def train_and_save_models(self):
        """Train maternal risk models with current XGBoost version"""
        try:
            print("Training maternal risk models...")
            
            # Load and preprocess data
            df = pd.read_csv('maternal_dataset.csv')
            print(f"Dataset shape: {df.shape}")
            
            # Map risk levels
            risk_level_mapping = {'low risk': 0, 'mid risk': 1, 'high risk': 2}
            df['RiskLevel'] = df['RiskLevel'].map(risk_level_mapping).astype(float)
            
            X = df.drop('RiskLevel', axis=1)
            y = df['RiskLevel']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )
            
            # Scale features
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Define models with current XGBoost syntax
            models_config = {
                'SVM': {
                    'model': SVC(class_weight='balanced', random_state=42, probability=True),
                    'params': {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto'], 'kernel': ['rbf', 'linear']}
                },
                                 'XGBoost': {
                     'model': xgb.XGBClassifier(
                         random_state=42, 
                         eval_metric='mlogloss',
                         enable_categorical=False,
                         use_label_encoder=False
                     ),
                     'params': {'n_estimators': [100, 200], 'max_depth': [3, 5], 'learning_rate': [0.1, 0.2]}
                 },
                'Random Forest': {
                    'model': RandomForestClassifier(class_weight='balanced', random_state=42),
                    'params': {'n_estimators': [100, 200], 'max_depth': [10, 20]}
                },
                'Decision Tree': {
                    'model': DecisionTreeClassifier(class_weight='balanced', random_state=42),
                    'params': {'max_depth': [10, 20], 'min_samples_split': [2, 5]}
                }
            }
            
            # Train models
            best_models = {}
            results = {}
            
            for model_name, config in models_config.items():
                print(f"Training {model_name}...")
                
                grid_search = GridSearchCV(
                    config['model'], 
                    config['params'], 
                    cv=3, 
                    scoring='accuracy',
                    n_jobs=-1
                )
                
                grid_search.fit(X_train_scaled, y_train)
                best_model = grid_search.best_estimator_
                
                # Evaluate
                test_acc = best_model.score(X_test_scaled, y_test)
                cv_scores = cross_val_score(best_model, X_train_scaled, y_train, cv=5)
                
                print(f"{model_name} - Test Accuracy: {test_acc:.4f}, CV: {cv_scores.mean():.4f}")
                
                best_models[model_name] = best_model
                results[model_name] = (test_acc, cv_scores.mean())
            
            # Create ensemble
            top_models = sorted(results.items(), key=lambda x: x[1][1], reverse=True)[:3]
            ensemble_estimators = [(name, best_models[name]) for name, _ in top_models]
            ensemble_model = VotingClassifier(estimators=ensemble_estimators, voting='soft')
            ensemble_model.fit(X_train_scaled, y_train)
            
            best_models['Ensemble'] = ensemble_model
            test_acc = ensemble_model.score(X_test_scaled, y_test)
            cv_scores = cross_val_score(ensemble_model, X_train_scaled, y_train, cv=5)
            results['Ensemble'] = (test_acc, cv_scores.mean())
            
            print(f"Ensemble - Test Accuracy: {test_acc:.4f}, CV: {cv_scores.mean():.4f}")
            
            # Find best model
            best_model_name = max(results.keys(), key=lambda x: results[x][1])
            print(f"Best model: {best_model_name}")
            
            # Save models
            if not os.path.exists('saved_models'):
                os.makedirs('saved_models')
            
            with open('saved_models/scaler.pkl', 'wb') as f:
                pickle.dump(self.scaler, f)
            
            for model_name, model in best_models.items():
                filename = f'saved_models/{model_name.lower().replace(" ", "_")}_model.pkl'
                with open(filename, 'wb') as f:
                    pickle.dump(model, f)
                print(f"Saved {model_name} model")
            
            # Save model info
            model_info = {
                'feature_names': list(X.columns),
                'class_mapping': {0: 'low risk', 1: 'mid risk', 2: 'high risk'},
                'reverse_mapping': risk_level_mapping,
                'best_model': best_model_name,
                'results': results
            }
            
            with open('saved_models/model_info.pkl', 'wb') as f:
                pickle.dump(model_info, f)
            
            # Update instance variables
            self.models = best_models
            self.model_info = model_info
            
            print("Maternal risk models retrained and saved successfully!")
            
        except Exception as e:
            print(f"Error retraining models: {e}")
            self.models = {}

print("Initializing predictors...")
maternal_predictor = MaternalRiskPredictor()

preterm_predictor = PretermBirthPredictor()
if not preterm_predictor.load_models():
    if os.path.exists('lab_results.csv'):
        print("Training preterm birth models...")
        preterm_predictor.train_and_save_models()
    else:
        print("Warning: lab_results.csv not found. Preterm birth prediction unavailable.")

@app.route('/')
def home():
    return render_template('index.html', 
                         feature_names=maternal_predictor.model_info['feature_names'] if maternal_predictor.model_info else [],
                         models=list(maternal_predictor.models.keys()) if maternal_predictor.models else [])

@app.route('/api/predict/maternal', methods=['POST'])
def predict_maternal():
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        required_features = maternal_predictor.model_info['feature_names']
        
        if 'features' in data:
            features = data['features']
            if len(features) != len(required_features):
                return jsonify({
                    "error": f"Expected {len(required_features)} features, got {len(features)}",
                    "required_features": required_features
                }), 400
        else:
            features = []
            for feature in required_features:
                if feature not in data:
                    return jsonify({
                        "error": f"Missing feature: {feature}",
                        "required_features": required_features
                    }), 400
                features.append(data[feature])
        
        model_name = data.get('model', None)
        result = maternal_predictor.predict_single(features, model_name)
        
        if "error" in result:
            return jsonify(result), 400
        
        response = {
            "success": True,
            "input_features": dict(zip(required_features, features)),
            "predictions": result,
            "prediction_type": "maternal_risk"
        }
        
        if maternal_predictor.model_info and 'best_model' in maternal_predictor.model_info:
            response["recommended_model"] = maternal_predictor.model_info['best_model']
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/api/predict/preterm', methods=['POST'])
def predict_preterm():
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        if not preterm_predictor.models:
            return jsonify({"error": "Preterm birth models not available"}), 503
        
        # Extract model name if specified
        model_name = data.pop('model', None) if 'model' in data else None
        
        result = preterm_predictor.predict_preterm_risk(data, model_name)
        
        if "error" in result:
            return jsonify(result), 400
        
        response = {
            "success": True,
            "input_data": data,
            "predictions": result,
            "prediction_type": "preterm_birth"
        }
        
        # Add recommended model info
        if preterm_predictor.best_model:
            response["recommended_model"] = preterm_predictor.best_model
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/api/predict/preterm/image', methods=['POST'])
def predict_preterm_from_image():
    try:
        # Check if image file is provided
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided. Please upload an image with 'image' field."}), 400
        
        image_file = request.files['image']
        if not image_file or not image_file.filename:
            return jsonify({"error": "Image file is invalid or empty"}), 400
        
        if not allowed_file(image_file.filename):
            return jsonify({"error": f"File type not allowed. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"}), 400
        
        if not preterm_predictor.models:
            return jsonify({"error": "Preterm birth models not available"}), 503
        
        # Process image to extract medical data
        extracted_data, status_code = process_medical_image(image_file, "preterm")
        
        if status_code != 200:
            return jsonify(extracted_data), status_code
        
        # Get model name from form data if specified (ensure empty string becomes None)
        model_name = request.form.get('model', None)
        if model_name == "" or model_name == "null":
            model_name = None
        
        # Clean extracted data (remove null values)
        clean_data = {k: v for k, v in extracted_data.items() if v is not None}
        
        if not clean_data:
            return jsonify({
                "error": "No valid medical data could be extracted from the image",
                "extracted_data": extracted_data
            }), 400
        
        # Make prediction
        result = preterm_predictor.predict_preterm_risk(clean_data, model_name)
        
        if "error" in result:
            return jsonify({
                "error": result["error"],
                "extracted_data": extracted_data
            }), 400
        
        response = {
            "success": True,
            "extraction_method": "image_analysis",
            "extracted_data": extracted_data,
            "used_data": clean_data,
            "predictions": result,
            "prediction_type": "preterm_birth"
        }
        
        # Add recommended model info
        if preterm_predictor.best_model:
            response["recommended_model"] = preterm_predictor.best_model
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/api/predict/maternal/image', methods=['POST'])
def predict_maternal_from_image():
    try:
        # Check if image file is provided
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided. Please upload an image with 'image' field."}), 400
        
        image_file = request.files['image']
        if not image_file or not image_file.filename:
            return jsonify({"error": "Image file is invalid or empty"}), 400
        
        if not allowed_file(image_file.filename):
            return jsonify({"error": f"File type not allowed. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"}), 400
        
        if not maternal_predictor.models:
            return jsonify({"error": "Maternal risk models not available"}), 503
        
        # Process image to extract medical data
        extracted_data, status_code = process_medical_image(image_file, "maternal")
        
        if status_code != 200:
            return jsonify(extracted_data), status_code
        
        # Get model name from form data if specified (ensure empty string becomes None)
        model_name = request.form.get('model', None)
        if model_name == "" or model_name == "null":
            model_name = None
        
        # For maternal risk, we need to convert to the expected format
        # This will depend on what features the maternal model expects
        required_features = maternal_predictor.model_info['feature_names'] if maternal_predictor.model_info else []
        
        if not required_features:
            return jsonify({"error": "Maternal risk model feature names not available"}), 503
        
        # Convert extracted data to feature array
        features = []
        used_data = {}
        
        for feature in required_features:
            if feature in extracted_data and extracted_data[feature] is not None:
                features.append(float(extracted_data[feature]))
                used_data[feature] = extracted_data[feature]
            else:
                # Use default/average values for missing features (could be improved)
                features.append(0.0)
        
        if len(features) != len(required_features):
            return jsonify({
                "error": f"Expected {len(required_features)} features, could only extract {len(features)}",
                "required_features": required_features,
                "extracted_data": extracted_data
            }), 400
        
        # Make prediction
        result = maternal_predictor.predict_single(features, model_name)
        
        if "error" in result:
            return jsonify({
                "error": result["error"],
                "extracted_data": extracted_data
            }), 400
        
        response = {
            "success": True,
            "extraction_method": "image_analysis",
            "extracted_data": extracted_data,
            "used_features": dict(zip(required_features, features)),
            "predictions": result,
            "prediction_type": "maternal_risk"
        }
        
        # Add recommended model info
        if maternal_predictor.model_info and 'best_model' in maternal_predictor.model_info:
            response["recommended_model"] = maternal_predictor.model_info['best_model']
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/api/models', methods=['GET'])
def get_models():
    response = {
        "maternal_risk": {
            "available": len(maternal_predictor.models) > 0,
            "models": {}
        },
        "preterm_birth": {
            "available": len(preterm_predictor.models) > 0,
            "models": {}
        }
    }
    
    if maternal_predictor.models:
        for name in maternal_predictor.models.keys():
            response["maternal_risk"]["models"][name] = {"name": name, "available": True}
        
        response["maternal_risk"]["feature_names"] = maternal_predictor.model_info['feature_names'] if maternal_predictor.model_info else []
        response["maternal_risk"]["class_mapping"] = maternal_predictor.model_info['class_mapping'] if maternal_predictor.model_info else {}
        response["maternal_risk"]["best_model"] = maternal_predictor.model_info.get('best_model', None) if maternal_predictor.model_info else None
    
    if preterm_predictor.models:
        for name, result in preterm_predictor.results.items():
            response["preterm_birth"]["models"][name] = {
                "name": name,
                "available": True,
                "test_accuracy": round(result['test_acc'], 4),
                "f1_score": round(result['f1_score'], 4),
                "precision": round(result['precision'], 4),
                "recall": round(result['recall'], 4)
            }
        
        response["preterm_birth"]["feature_names"] = preterm_predictor.feature_names
        response["preterm_birth"]["class_mapping"] = {0: 'Term Birth', 1: 'Preterm Birth'}
        response["preterm_birth"]["best_model"] = preterm_predictor.best_model
    
    return jsonify(response)

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "maternal_risk_models": len(maternal_predictor.models),
        "preterm_birth_models": len(preterm_predictor.models),
        "gemini_configured": gemini_model_client is not None,
        "services": {
            "maternal_risk_prediction": len(maternal_predictor.models) > 0,
            "preterm_birth_prediction": len(preterm_predictor.models) > 0,
            "image_processing": gemini_model_client is not None
        },
        "endpoints": {
            "maternal_risk_prediction": "/api/predict/maternal",
            "maternal_risk_image_prediction": "/api/predict/maternal/image",
            "preterm_birth_prediction": "/api/predict/preterm",
            "preterm_birth_image_prediction": "/api/predict/preterm/image",
            "models_info": "/api/models",
            "health_check": "/api/health"
        }
    })

if __name__ == '__main__':
    print("="*60)
    print("INTEGRATED MATERNAL HEALTH PREDICTION API")
    print("="*60)
    print(f"Maternal Risk Models: {len(maternal_predictor.models)}")
    print(f"Preterm Birth Models: {len(preterm_predictor.models)}")
    print(f"Gemini Image Processing: {'✓ Enabled' if gemini_model_client else '✗ Disabled'}")
    print("\nAvailable endpoints:")
    print("- GET  /: Home page")
    print("- POST /api/predict/maternal: Maternal risk prediction (JSON)")
    print("- POST /api/predict/maternal/image: Maternal risk prediction (Image)")
    print("- POST /api/predict/preterm: Preterm birth prediction (JSON)")
    print("- POST /api/predict/preterm/image: Preterm birth prediction (Image)")
    print("- GET  /api/models: Get all model information")
    print("- GET  /api/health: Health check")
    print("\nImage Endpoints Support:")
    print("- Medical reports, lab results, prescription images")
    print("- Automatic data extraction using Gemini + OCR")
    print("- Supported formats: PNG, JPG, JPEG, TIFF, BMP, WEBP, PDF")
    print("="*60)
    
    app.run(debug=True, host='0.0.0.0', port=5000) 