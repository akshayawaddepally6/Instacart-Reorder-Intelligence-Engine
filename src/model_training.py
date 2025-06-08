"""
Model Training - SIMPLIFIED WORKING VERSION
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

def main():
    print("="*60)
    print("INSTACART MODEL TRAINING - SIMPLIFIED")
    print("="*60)
    
    # Load processed data
    print("\n1. Loading processed data...")
    data = pd.read_csv('data/processed/processed_data.csv')
    print(f"✓ Loaded {len(data):,} records")
    print(f"✓ Columns: {data.shape[1]}")
    
    # Check for any non-numeric columns
    print("\n2. Verifying data types...")
    print("Column types:")
    print(data.dtypes)
    
    # Separate features and target
    print("\n3. Preparing features and target...")
    X = data.drop('reordered', axis=1)
    y = data['reordered']
    
    print(f"✓ Features shape: {X.shape}")
    print(f"✓ Target shape: {y.shape}")
    print(f"✓ Reorder rate: {y.mean():.2%}")
    
    # Train-test split
    print("\n4. Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"✓ Train: {len(X_train):,} samples")
    print(f"✓ Test: {len(X_test):,} samples")
    
    # Train Random Forest
    print("\n" + "="*60)
    print("TRAINING RANDOM FOREST")
    print("="*60)
    
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=20,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    print("Training Random Forest...")
    rf.fit(X_train, y_train)
    
    # Evaluate Random Forest
    y_pred_rf = rf.predict(X_test)
    y_pred_proba_rf = rf.predict_proba(X_test)[:, 1]
    
    print("\nRandom Forest Results:")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred_rf):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred_rf):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred_rf):.4f}")
    print(f"F1 Score:  {f1_score(y_test, y_pred_rf):.4f}")
    print(f"ROC-AUC:   {roc_auc_score(y_test, y_pred_proba_rf):.4f}")
    
    # Train XGBoost
    print("\n" + "="*60)
    print("TRAINING XGBOOST")
    print("="*60)
    
    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=7,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss',
        verbosity=1
    )
    
    print("Training XGBoost...")
    xgb.fit(X_train, y_train)
    
    # Evaluate XGBoost
    y_pred_xgb = xgb.predict(X_test)
    y_pred_proba_xgb = xgb.predict_proba(X_test)[:, 1]
    
    print("\nXGBoost Results:")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred_xgb):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred_xgb):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred_xgb):.4f}")
    print(f"F1 Score:  {f1_score(y_test, y_pred_xgb):.4f}")
    print(f"ROC-AUC:   {roc_auc_score(y_test, y_pred_proba_xgb):.4f}")
    
    # Compare models
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    
    rf_auc = roc_auc_score(y_test, y_pred_proba_rf)
    xgb_auc = roc_auc_score(y_test, y_pred_proba_xgb)
    
    print(f"Random Forest ROC-AUC: {rf_auc:.4f}")
    print(f"XGBoost ROC-AUC:       {xgb_auc:.4f}")
    
    if xgb_auc > rf_auc:
        best_model = xgb
        best_name = "XGBoost"
        best_auc = xgb_auc
    else:
        best_model = rf
        best_name = "Random Forest"
        best_auc = rf_auc
    
    print(f"\n🏆 Best Model: {best_name} (ROC-AUC: {best_auc:.4f})")
    
    # Feature importance
    print("\n" + "="*60)
    print("TOP 10 FEATURE IMPORTANCE")
    print("="*60)
    
    feature_names = X_train.columns
    importance = best_model.feature_importances_
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False).head(10)
    
    print(importance_df.to_string(index=False))
    
    # Save model
    print("\n" + "="*60)
    print("SAVING MODEL")
    print("="*60)
    
    Path('models').mkdir(exist_ok=True)
    joblib.dump(best_model, 'models/reorder_model.pkl')
    joblib.dump(feature_names.tolist(), 'models/feature_names.pkl')
    
    print(f"✓ Saved {best_name} to models/reorder_model.pkl")
    print(f"✓ Saved feature names to models/feature_names.pkl")
    
    print("\n" + "="*60)
    print("✓ TRAINING COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    main()
