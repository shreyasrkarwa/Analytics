import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

def load_data():
    df = pd.read_csv('static_data.csv')
    X = df.drop(['account_id', 'time_to_event', 'event_observed'], axis=1)
    y = df['event_observed']
    return X, y

def train_baselines(X, y):
    numeric_features = ['initial_arr', 'onboarding_duration_days']
    categorical_features = ['industry', 'account_segment', 'account_region', 'contract_length_months', 'has_channel_partner']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Logistic Regression
    lr_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('classifier', LogisticRegression(max_iter=1000))])
    lr_pipeline.fit(X_train, y_train)
    lr_preds = lr_pipeline.predict(X_test)
    lr_probs = lr_pipeline.predict_proba(X_test)[:, 1]
    
    print("=== Baseline: Logistic Regression ===")
    print(f"ROC-AUC: {roc_auc_score(y_test, lr_probs):.4f}")
    
    # Random Forest
    rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))])
    rf_pipeline.fit(X_train, y_train)
    rf_preds = rf_pipeline.predict(X_test)
    rf_probs = rf_pipeline.predict_proba(X_test)[:, 1]
    
    print("\n=== Baseline: Random Forest ===")
    print(f"ROC-AUC: {roc_auc_score(y_test, rf_probs):.4f}")
    
    return lr_pipeline, rf_pipeline

if __name__ == "__main__":
    X, y = load_data()
    train_baselines(X, y)
