# META DATA - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

    # Developer details: 
        # Name: Vansh R
        # Role: Architect
        # Code ownership rights: Vansh R
    # Version:
        # Version: V 1.0 (29 July 2024)
            # Developer: Vansh R
            # Unit test: Pass
            # Integration test: Pass
     
# CODE - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

# Dependency: 
    # Environment:     
        # Redis: 5.0.7
        # Pandas: 2.2.2
        # Scikit-learn: 1.5.0

# Import necessary libraries
import redis
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix

def load_data_from_redis(r, key):
    # Load and deserialize data from Redis using the provided key
    data = pickle.loads(r.get(key))
    # Return a DataFrame if the key indicates features, otherwise return a Series
    return pd.DataFrame(data) if 'X_' in key else pd.Series(data)

def evaluate_test_data(X_test, y_test, model):
    # Predict labels for the test set
    y_pred_test = model.predict(X_test)
    # Calculate accuracy score for the test set
    test_accuracy = accuracy_score(y_test, y_pred_test)
    # Calculate ROC AUC score for the test set
    if hasattr(model, "predict_proba"):
        test_roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    else:
        test_roc_auc = None
    # Return accuracy, ROC AUC, confusion matrix, and classification report for the test set
    return test_accuracy, test_roc_auc, confusion_matrix(y_test, y_pred_test), classification_report(y_test, y_pred_test)

def evaluate_validation_data(X_val, y_val, model):
    # Predict labels for the validation set
    y_pred_val = model.predict(X_val)
    # Calculate accuracy score for the validation set
    val_accuracy = accuracy_score(y_val, y_pred_val)
    # Calculate ROC AUC score for the validation set
    if hasattr(model, "predict_proba"):
        val_roc_auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
    else:
        val_roc_auc = None
    # Return accuracy, ROC AUC, confusion matrix, and classification report for the validation set
    return val_accuracy, val_roc_auc, confusion_matrix(y_val, y_pred_val), classification_report(y_val, y_pred_val)

def evaluate_supervalidation_data(X_superval, y_superval, model):
    # Predict labels for the super validation set
    y_pred_superval = model.predict(X_superval)
    # Calculate accuracy score for the super validation set
    superval_accuracy = accuracy_score(y_superval, y_pred_superval)
    # Calculate ROC AUC score for the super validation set
    if hasattr(model, "predict_proba"):
        superval_roc_auc = roc_auc_score(y_superval, model.predict_proba(X_superval)[:, 1])
    else:
        superval_roc_auc = None
    # Return accuracy, ROC AUC, confusion matrix, and classification report for the super validation set
    return superval_accuracy, superval_roc_auc, confusion_matrix(y_superval, y_pred_superval), classification_report(y_superval, y_pred_superval)

def evaluate_model(redis_host, redis_port, redis_db, model_path):
    
    # Connect to Redis database
    r = redis.Redis(host=redis_host, port=redis_port, db=redis_db)
    
    # Load test, validation, and super validation data from Redis
    X_test = load_data_from_redis(r, 'X_test')
    y_test = load_data_from_redis(r, 'y_test')
    X_val = load_data_from_redis(r, 'X_val')
    y_val = load_data_from_redis(r, 'y_val')
    X_superval = load_data_from_redis(r, 'X_superval')
    y_superval = load_data_from_redis(r, 'y_superval')
    
    # Ensure column names are strings for consistency
    X_test = X_test.rename(str, axis="columns")
    X_val = X_val.rename(str, axis="columns")
    X_superval = X_superval.rename(str, axis="columns")

    # Load the best model from the pickle file
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # Evaluate the model on test data
    test_accuracy, test_roc_auc, test_confusion_matrix, test_classification_report = evaluate_test_data(X_test, y_test, model)
    # Evaluate the model on validation data
    val_accuracy, val_roc_auc, val_confusion_matrix, val_classification_report = evaluate_validation_data(X_val, y_val, model)
    # Evaluate the model on super validation data
    superval_accuracy, superval_roc_auc, superval_confusion_matrix, superval_classification_report = evaluate_supervalidation_data(X_superval, y_superval, model)
    
    # Return evaluation metrics for test, validation, and super validation data
    return test_accuracy, test_roc_auc, test_confusion_matrix, test_classification_report, val_accuracy, val_roc_auc, val_confusion_matrix, val_classification_report, superval_accuracy, superval_roc_auc, superval_confusion_matrix, superval_classification_report

if __name__ == "__main__":
    test_accuracy, test_roc_auc, test_confusion_matrix, test_classification_report, val_accuracy, val_roc_auc, val_confusion_matrix, val_classification_report, superval_accuracy, superval_roc_auc, superval_confusion_matrix, superval_classification_report = evaluate_model('localhost', 6379, 1, 'bagging_model.pkl')
