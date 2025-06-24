from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd
import os
from flask_cors import CORS
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

app = Flask(__name__)
CORS(app)

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
                'model': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
                'params': {'n_estimators': [100, 200], 'max_depth': [3, 5], 'learning_rate': [0.1, 0.2]}
            },
            'CatBoost': {
                'model': CatBoostClassifier(random_state=42, verbose=False, class_weights='Balanced'),
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
                    with open(filename, 'rb') as f:
                        self.models[model_name] = pickle.load(f)
            
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
        "services": {
            "maternal_risk_prediction": len(maternal_predictor.models) > 0,
            "preterm_birth_prediction": len(preterm_predictor.models) > 0
        }
    })

if __name__ == '__main__':
    print("="*60)
    print("INTEGRATED MATERNAL HEALTH PREDICTION API")
    print("="*60)
    print(f"Maternal Risk Models: {len(maternal_predictor.models)}")
    print(f"Preterm Birth Models: {len(preterm_predictor.models)}")
    print("\nAvailable endpoints:")
    print("- GET  /: Home page")
    print("- POST /api/predict/maternal: Maternal risk prediction")
    print("- POST /api/predict/preterm: Preterm birth prediction")
    print("- GET  /api/models: Get all model information")
    print("- GET  /api/health: Health check")
    print("="*60)
    
    app.run(debug=True, host='0.0.0.0', port=5000) 