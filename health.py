import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                           confusion_matrix, classification_report, roc_auc_score, roc_curve)
import xgboost as xgb
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
import pickle
import warnings
import os

warnings.filterwarnings('ignore')

class PretermBirthPredictor:
    def __init__(self, data_path='lab_results.csv'):
        self.data_path = data_path
        self.models = {}
        self.scaler = StandardScaler()
        self.best_model = None
        self.feature_names = None
        self.results = {}
        
    def load_and_preprocess_data(self):
        print("Loading and preprocessing data...")
        
        df = pd.read_csv(self.data_path, sep=';')
        print(f"Dataset shape: {df.shape}")
        print(f"Missing values per column:\n{df.isnull().sum()}")
        
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
        print(f"Class distribution:\n{df['Label [0-term. 1-preterm]'].value_counts()}")
        
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
        print(f"Features used: {self.feature_names}")
        
        return X, y
    
    def split_and_scale_data(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
        print(f"Training class distribution: {y_train.value_counts().to_dict()}")
        print(f"Test class distribution: {y_test.value_counts().to_dict()}")
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def evaluate_model(self, model, X_train, X_test, y_train, y_test, model_name):
        print(f"\nEvaluating {model_name}...")
        
        model.fit(X_train, y_train)
        
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        train_acc = accuracy_score(y_train, y_pred_train)
        test_acc = accuracy_score(y_test, y_pred_test)
        precision = precision_score(y_test, y_pred_test)
        recall = recall_score(y_test, y_pred_test)
        f1 = f1_score(y_test, y_pred_test)
        
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)[:, 1]
            auc_score = roc_auc_score(y_test, y_proba)
        else:
            auc_score = None
        
        print(f"=== {model_name} Results ===")
        print(f"Training Accuracy: {train_acc:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"Cross-Validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        if auc_score:
            print(f"AUC-ROC: {auc_score:.4f}")
        
        print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred_test)}")
        print(f"Classification Report:\n{classification_report(y_test, y_pred_test)}")
        
        return {
            'model': model,
            'train_acc': train_acc,
            'test_acc': test_acc,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'auc_score': auc_score,
            'predictions': y_pred_test
        }
    
    def train_models(self, X_train, X_test, y_train, y_test):
        print("\n" + "="*60)
        print("TRAINING MACHINE LEARNING MODELS")
        print("="*60)
        
        models_config = {
            'Logistic Regression': {
                'model': LogisticRegression(random_state=42, class_weight='balanced'),
                'params': {
                    'C': [0.01, 0.1, 1, 10, 100],
                    'solver': ['liblinear', 'lbfgs'],
                    'penalty': ['l1', 'l2']
                }
            },
            'Linear SVM': {
                'model': SVC(kernel='linear', random_state=42, class_weight='balanced', probability=True),
                'params': {
                    'C': [0.01, 0.1, 1, 10, 100],
                    'gamma': ['scale', 'auto']
                }
            },
            'RBF SVM': {
                'model': SVC(kernel='rbf', random_state=42, class_weight='balanced', probability=True),
                'params': {
                    'C': [0.1, 1, 10, 100],
                    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]
                }
            },
            'Random Forest': {
                'model': RandomForestClassifier(random_state=42, class_weight='balanced'),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 5, 10, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            },
            'Decision Tree': {
                'model': DecisionTreeClassifier(random_state=42, class_weight='balanced'),
                'params': {
                    'max_depth': [3, 5, 10, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'criterion': ['gini', 'entropy']
                }
            },
            'XGBoost': {
                'model': xgb.XGBClassifier(
                    random_state=42, 
                    eval_metric='logloss',
                    enable_categorical=False,
                    use_label_encoder=False
                ),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [3, 4, 5, 6],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.8, 0.9, 1.0]
                }
            },
            'CatBoost': {
                'model': CatBoostClassifier(random_state=42, verbose=False, class_weights='Balanced'),
                'params': {
                    'iterations': [100, 200],
                    'depth': [4, 6, 8],
                    'learning_rate': [0.03, 0.1, 0.2],
                    'l2_leaf_reg': [1, 3, 5]
                }
            }
        }
        
        for model_name, config in models_config.items():
            print(f"\nOptimizing {model_name}...")
            
            grid_search = GridSearchCV(
                config['model'], 
                config['params'], 
                cv=5, 
                scoring='f1',
                n_jobs=-1
            )
            
            grid_search.fit(X_train, y_train)
            
            print(f"Best parameters for {model_name}: {grid_search.best_params_}")
            
            best_model = grid_search.best_estimator_
            result = self.evaluate_model(best_model, X_train, X_test, y_train, y_test, model_name)
            
            self.models[model_name] = best_model
            self.results[model_name] = result
    
    def create_ensemble(self, X_train, X_test, y_train, y_test):
        print("\nCreating Ensemble Model...")
        
        top_models = sorted(self.results.items(), key=lambda x: x[1]['f1_score'], reverse=True)[:3]
        print(f"Top 3 models for ensemble: {[name for name, _ in top_models]}")
        
        ensemble_estimators = [(name, self.models[name]) for name, _ in top_models]
        
        ensemble_model = VotingClassifier(estimators=ensemble_estimators, voting='soft')
        
        result = self.evaluate_model(ensemble_model, X_train, X_test, y_train, y_test, "Ensemble")
        
        self.models['Ensemble'] = ensemble_model
        self.results['Ensemble'] = result
    
    def display_results_summary(self):
        print("\n" + "="*80)
        print("FINAL RESULTS SUMMARY")
        print("="*80)
        
        summary_data = []
        for model_name, result in self.results.items():
            summary_data.append({
                'Model': model_name,
                'Test Accuracy': f"{result['test_acc']:.4f}",
                'Precision': f"{result['precision']:.4f}",
                'Recall': f"{result['recall']:.4f}",
                'F1-Score': f"{result['f1_score']:.4f}",
                'CV Score': f"{result['cv_mean']:.4f}",
                'AUC-ROC': f"{result['auc_score']:.4f}" if result['auc_score'] else "N/A"
            })
        
        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))
        
        best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['f1_score'])
        self.best_model = best_model_name
        
        print(f"\nBest performing model: {best_model_name}")
        print(f"F1-Score: {self.results[best_model_name]['f1_score']:.4f}")
        
        best_result = self.results[best_model_name]
        print(f"\nBest Model Performance Summary:")
        print(f"- Accuracy: {best_result['test_acc']:.1%}")
        print(f"- Precision: {best_result['precision']:.1%}")
        print(f"- Recall: {best_result['recall']:.1%}")
        print(f"- F1-Score: {best_result['f1_score']:.1%}")
        if best_result['auc_score']:
            print(f"- AUC-ROC: {best_result['auc_score']:.1%}")
    
    def save_models(self):
        print("\n" + "="*60)
        print("SAVING MODELS AND METADATA")
        print("="*60)
        
        if not os.path.exists('saved_models'):
            os.makedirs('saved_models')
        
        with open('saved_models/preterm_birth_scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        print("Saved preterm birth scaler")
        
        for model_name, model in self.models.items():
            filename = f"saved_models/preterm_birth_{model_name.lower().replace(' ', '_')}_model.pkl"
            with open(filename, 'wb') as f:
                pickle.dump(model, f)
            print(f"Saved preterm birth {model_name} model")
        
        model_info = {
            'feature_names': self.feature_names,
            'best_model': self.best_model,
            'results': {name: {k: v for k, v in result.items() if k != 'model'} 
                       for name, result in self.results.items()},
            'class_mapping': {0: 'Term Birth', 1: 'Preterm Birth'}
        }
        
        with open('saved_models/preterm_birth_model_info.pkl', 'wb') as f:
            pickle.dump(model_info, f)
        print("Saved preterm birth model metadata")
    
    def predict_preterm_risk(self, patient_data):
        if self.best_model is None:
            raise ValueError("No trained model available. Please train the model first.")
        
        if isinstance(patient_data, dict):
            patient_df = pd.DataFrame([patient_data])
        else:
            patient_df = patient_data.copy()
        
        patient_df = self.feature_engineering(patient_df)
        
        patient_features = patient_df[self.feature_names]
        patient_scaled = self.scaler.transform(patient_features)
        
        model = self.models[self.best_model]
        prediction = model.predict(patient_scaled)[0]
        
        if hasattr(model, 'predict_proba'):
            probability = model.predict_proba(patient_scaled)[0]
            risk_probability = probability[1]
        else:
            risk_probability = None
        
        result = {
            'prediction': 'Preterm Birth' if prediction == 1 else 'Term Birth',
            'risk_score': prediction,
            'probability': risk_probability,
            'model_used': self.best_model
        }
        
        return result
    
    def run_complete_analysis(self):
        print("Starting Preterm Birth Prediction Analysis")
        print("Based on: 'Predicting preterm birth using machine learning methods'")
        print("Reference: https://www.nature.com/articles/s41598-025-89905-1")
        print("="*80)
        
        df = self.load_and_preprocess_data()
        
        df = self.feature_engineering(df)
        
        X, y = self.prepare_features(df)
        
        X_train, X_test, y_train, y_test = self.split_and_scale_data(X, y)
        
        self.train_models(X_train, X_test, y_train, y_test)
        
        self.create_ensemble(X_train, X_test, y_train, y_test)
        
        self.display_results_summary()
        
        self.save_models()
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE!")
        print("="*80)
        print("Key findings from the research paper:")
        print("- Linear SVM achieved 82% accuracy, 83% precision, 86% recall, 84% F1-score")
        print("- Logistic Regression also performed well with 80% accuracy")
        print("- Important features: week of delivery, gestational diabetes, education level")
        print("\nYour model is now ready for preterm birth risk prediction!")

def main():
    predictor = PretermBirthPredictor('lab_results.csv')
    predictor.run_complete_analysis()
    
    print("\nExample prediction:")
    sample_patient = {
        'Hb [g/dl]': 12.5,
        'WBC [G/l]': 11.25,
        'PLT [G/l]': 261,
        'HCT [%]': 35.8,
        'Age': 31,
        'No. of pregnancy': 1,
        'No. of deliveries': 1,
        'Week of sample collection': 32,
        'Height': 171,
        'Weight': 64,
        'BMI': 21.9,
        'Education [0-primary. 1-vocational. 2-higher]': 2,
        'Marital status [0-single. 1-married]': 1,
        'Gestational diabetes mellitus [0-no. 1-type 1. 2-type 2]': 0,
        'Gestational hypothyroidism [0-no.-1yes]': 0,
        'History of preterm labour [0-no.1-yes]': 0,
        'Smoking [0-no.1-yes]': 0,
        'History of surgical delivery [0-no.1-yes]': 0,
        'History of caesarean section [0-no.1-yes]': 0,
        'CRP': 4.7,
        'Week of delivery': 31,
        'Type of delivery [0-vaginal.1-c-section]': 0
    }
    
    try:
        prediction = predictor.predict_preterm_risk(sample_patient)
        print(f"Prediction: {prediction}")
    except Exception as e:
        print(f"Prediction error: {e}")

if __name__ == "__main__":
    main()
