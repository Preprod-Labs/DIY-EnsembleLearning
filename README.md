# DIY-EnsembleLearning

This is the Ensemble Learning Algorithms branch.

# Ensemble Learning Algorithms

### - Bagging
Bagging (Bootstrap Aggregating) is an ensemble learning method that improves the stability and accuracy of machine learning algorithms by training multiple models on different random subsets of the training data (with replacement). The final prediction is obtained by averaging (for regression) or majority voting (for classification) of these individual models.

### - Stacking Classifier
Stacking is an ensemble learning technique that combines multiple base models (of different types) and a meta-model. The base models are trained on the same dataset, and their predictions are used as inputs for the meta-model, which then makes the final prediction. This method leverages the strengths of diverse models to improve overall performance.

### - Voting Classifier
Voting classifiers aggregate the predictions of multiple base models to improve overall classification performance. There are two types: hard voting, where the final class label is determined by the majority vote of the base models, and soft voting, where the class probabilities predicted by each model are averaged, and the class with the highest average probability is selected.

## Problem Definition

The business provides a subscription-based service (e.g., streaming platform, SaaS product, etc.). The goal is to use the Bagging Algorithm to predict customer churn, i.e., whether a customer is likely to cancel their subscription.

## Data Definition

Mock data for learning purposes with features: customer_id, signup_date, last_login, annual_fee, subscription_type, payment_method account_age, number_of_logins, total_spent, num_tickets_raised, avg_response_time, satisfaction_score, country, device, usage_hours_per_month, churn

> **Note:** The dataset consists of 1000 samples, leading to potential overfitting with a high training accuracy. This would not occur in real-life scenarios with larger and more varied datasets, providing a more realistic accuracy.

## Directory Structure

- **Code/**: Contains all the scripts for data ingestion, transformation, loading, evaluation, model training, inference, manual prediction, and web application.
- **Data/**: Contains the raw mock data.

#### Data Splitting

- **Training Samples**: 600
- **Testing Samples**: 150
- **Validation Samples**: 150
- **Supervalidation Samples**: 100

# Program Flow

1. **`db_utils`:** This code snippet contains utility functions to connect to PostgreSQL and Cassandra databases, create tables, and insert data into them.
2. **Data Ingestion:** This code snippet ingests customer data from a CSV file, preprocesses it, and stores it in PostgreSQL and Cassandra databases. [`ingest.py`]
3. **Data Preprocessing:** This code snippet contains utility functions to evaluate a model using test, validation, and super validation data stored in a Redis database. [`preprocess.py`]
4. **Data Splitting:** This code snippet preprocesses input data for a machine learning model by scaling numerical columns, encoding categorical columns, and extracting date components for further analysis. [`split.py`]
5. **Model Training:** This is where Bagging, Stacking, and Voting Classifier models, using the training data, are trained and stored in a Redis database. [`train_bagging.py`, `train_stacking_classifier.py`, `train_voting_classifier.py`]
6. **Model Evaluation:** This code snippet contains utility functions to evaluate a model using test, validation, and super validation data stored in a Redis database. [`model_eval.py`]
7. **Model Prediction:** This code snippet preprocesses input data for a machine learning model by scaling numerical columns, encoding categorical columns, and extracting date components for further analysis. [`model_predict.py`]
8. **Web Application:** This code snippet creates a web app using Streamlit to train, evaluate, and predict churn using three different ensemble models: Bagging, Voting Classifier, and Stacking Classifier. [`app.py`]

## Steps to Run

1. Install the necessary packages: `pip install -r requirements.txt`
2. Run the Streamlit web application: `streamlit run Code/app.py`