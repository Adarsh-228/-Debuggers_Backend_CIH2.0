from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn import tree
import xgboost as xgb
from sklearn.svm import SVC
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
import warnings
warnings.filterwarnings('ignore')

print("Loading and preprocessing data...")
df = pd.read_csv('./maternal_dataset.csv')

print(f"Dataset shape: {df.shape}")
print(f"Class distribution:\n{df['RiskLevel'].value_counts()}")

RiskLevel = {'low risk': 0, 'mid risk': 1, 'high risk': 2}
df['RiskLevel'] = df['RiskLevel'].map(RiskLevel).astype(float)

X = df.drop('RiskLevel', axis=1)
y = df['RiskLevel']

print(f"Features: {list(X.columns)}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

ss = StandardScaler()
X_train_scaled = ss.fit_transform(X_train)
X_test_scaled = ss.transform(X_test)

class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
print(f"Class weights: {class_weight_dict}")

def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    model.fit(X_train, y_train)
    
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    
    y_pred = model.predict(X_test)
    
    print(f"\n=== {model_name} ===")
    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    print(f"Classification Report:\n{classification_report(y_test, y_pred)}")
    
    return test_acc, cv_scores.mean()

models = {}
results = {}

print("\n" + "="*50)
print("OPTIMIZED MODELS WITH HYPERPARAMETER TUNING")
print("="*50)

print("\n1. Optimizing SVM...")
svm_params = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
    'kernel': ['rbf', 'linear', 'poly']
}
svm_grid = GridSearchCV(SVC(class_weight='balanced', random_state=42), svm_params, cv=5, n_jobs=-1)
svm_grid.fit(X_train_scaled, y_train)
best_svm = svm_grid.best_estimator_
print(f"Best SVM params: {svm_grid.best_params_}")
models['SVM'] = best_svm
results['SVM'] = evaluate_model(best_svm, X_train_scaled, X_test_scaled, y_train, y_test, "Optimized SVM")

print("\n2. Optimizing XGBoost...")
xgb_params = {
    'n_estimators': [100, 200],
    'max_depth': [3, 4, 5, 6],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 0.9, 1.0]
}
xgb_grid = GridSearchCV(xgb.XGBClassifier(random_state=42), xgb_params, cv=5, n_jobs=-1)
xgb_grid.fit(X_train_scaled, y_train)
best_xgb = xgb_grid.best_estimator_
print(f"Best XGBoost params: {xgb_grid.best_params_}")
models['XGBoost'] = best_xgb
results['XGBoost'] = evaluate_model(best_xgb, X_train_scaled, X_test_scaled, y_train, y_test, "Optimized XGBoost")

print("\n3. Optimizing Random Forest...")
rf_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
rf_grid = GridSearchCV(RandomForestClassifier(class_weight='balanced', random_state=42), rf_params, cv=5, n_jobs=-1)
rf_grid.fit(X_train_scaled, y_train)
best_rf = rf_grid.best_estimator_
print(f"Best Random Forest params: {rf_grid.best_params_}")
models['Random Forest'] = best_rf
results['Random Forest'] = evaluate_model(best_rf, X_train_scaled, X_test_scaled, y_train, y_test, "Optimized Random Forest")

print("\n4. Optimizing Decision Tree...")
dt_params = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
dt_grid = GridSearchCV(tree.DecisionTreeClassifier(class_weight='balanced', random_state=42), dt_params, cv=5, n_jobs=-1)
dt_grid.fit(X_train_scaled, y_train)
best_dt = dt_grid.best_estimator_
print(f"Best Decision Tree params: {dt_grid.best_params_}")
models['Decision Tree'] = best_dt
results['Decision Tree'] = evaluate_model(best_dt, X_train_scaled, X_test_scaled, y_train, y_test, "Optimized Decision Tree")

print("\n5. Creating Ensemble Model...")
voting_clf = VotingClassifier(
    estimators=[
        ('svm', best_svm),
        ('xgb', best_xgb), 
        ('rf', best_rf)
    ],
    voting='hard'
)
models['Ensemble'] = voting_clf
results['Ensemble'] = evaluate_model(voting_clf, X_train_scaled, X_test_scaled, y_train, y_test, "Ensemble (Voting)")

print("\n" + "="*50)
print("FINAL RESULTS SUMMARY")
print("="*50)

best_model = None
best_score = 0
for model_name, (test_acc, cv_acc) in results.items():
    print(f"{model_name:15} | Test: {test_acc:.4f} | CV: {cv_acc:.4f}")
    if cv_acc > best_score:
        best_score = cv_acc
        best_model = model_name

print(f"\nBest performing model: {best_model} (CV Score: {best_score:.4f})")

print("\n" + "="*50)
print("RECOMMENDATIONS:")
print("="*50)
print("1. Use the best performing model above")
print("2. Consider feature engineering (age groups, BP categories)")
print("3. Collect more data if possible")
print("4. Try advanced ensemble methods (Stacking, Boosting)")
print("5. Consider deep learning for larger datasets")
