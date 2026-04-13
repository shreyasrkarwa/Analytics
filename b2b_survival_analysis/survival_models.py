import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv

def load_survival_data():
    df = pd.read_csv('static_data.csv')
    
    # sksurv requires y to be a structured array of (boolean status, time)
    y = Surv.from_dataframe('event_observed', 'time_to_event', df)
    
    X = df.drop(['account_id', 'time_to_event', 'event_observed'], axis=1)
    return X, y

def train_survival_models(X, y):
    numeric_features = ['initial_arr', 'onboarding_duration_days']
    categorical_features = ['industry', 'account_segment', 'account_region', 'contract_length_months', 'has_channel_partner']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ])
    
    X_transformed = preprocessor.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)
    
    # Cox Proportional Hazards
    cox_model = CoxPHSurvivalAnalysis(alpha=0.01)
    cox_model.fit(X_train, y_train)
    cox_c_index = cox_model.score(X_test, y_test)
    print("=== Cox Proportional Hazards ===")
    print(f"Concordance Index (C-Index): {cox_c_index:.4f}")
    
    # Random Survival Forest
    rsf_model = RandomSurvivalForest(n_estimators=100, min_samples_split=10, min_samples_leaf=15, max_features="sqrt", random_state=42)
    rsf_model.fit(X_train, y_train)
    rsf_c_index = rsf_model.score(X_test, y_test)
    print("\n=== Random Survival Forest ===")
    print(f"Concordance Index (C-Index): {rsf_c_index:.4f}")
    
    return cox_model, rsf_model, preprocessor

if __name__ == "__main__":
    X, y = load_survival_data()
    train_survival_models(X, y)
