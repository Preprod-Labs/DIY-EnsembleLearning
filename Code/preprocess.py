# META DATA - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

    # Developer details: 
        # Name: Mohini T and Vansh R
        # Role: Architects
        # Code ownership rights: Mohini T and Vansh R
    # Version:
        # Version: V 1.0 (11 July 2024)
            # Developers: Mohini T and Vansh R
            # Unit test: Pass
            # Integration test: Pass
     
    # Description: This code snippet contains utility functions to evaluate a model using test, validation,
    # and super validation data stored in a Redis database.
        # PostgreSQL: Yes
        # Cassandra: Yes

# CODE - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

    # Dependency: 
        # Environment:     
            # Python 3.11.5
            # Pandas 2.2.2
            # Scikit-learn 1.5.0

import pandas as pd                                             # Importing pandas for data manipulation
from sklearn.preprocessing import StandardScaler, LabelEncoder  # Importing tools for data preprocessing
import db_utils                                                 # Importing utility functions for database operations

def preprocess_postgres_data(data):
    # Separate customer_id
    customer_id = data['customer_id']

    # Define columns to be scaled, excluding 'customer_id'
    numerical_cols = [
        'annual_fee',
        'account_age',
        'number_of_logins',
        'total_spent',
        'num_tickets_raised',
        'avg_response_time',
        'satisfaction_score'
    ]

    # Create a temporary DataFrame for scaling
    temp_data = data[numerical_cols].copy()

    scaler = StandardScaler() # Initialize the StandardScaler
    temp_data = pd.DataFrame(scaler.fit_transform(temp_data), columns=numerical_cols) # Scale numerical columns

    # Encode categorical columns
    encoder = LabelEncoder() # Initialize the LabelEncoder
    categorical_cols = data.select_dtypes(include=['object']).columns.tolist() # Get list of categorical columns
    for col in categorical_cols:
        data[col] = encoder.fit_transform(data[col]) # Encode categorical columns

    # Rejoin customer_id and scaled numerical columns
    data = data.drop(columns=numerical_cols) # Drop original numerical columns
    data = pd.concat([data, temp_data], axis=1) # Concatenate scaled numerical columns back
    data['customer_id'] = customer_id # Reassign customer_id

    return data

def preprocess_cassandra_data(data):
    # Separate customer_id
    customer_id = data['customer_id']
    
    # Convert Cassandra date columns to strings
    data['last_login'] = data['last_login'].astype(str)
    data['signup_date'] = data['signup_date'].astype(str)
    
    # Convert string dates to datetime
    data['last_login'] = pd.to_datetime(data['last_login'])
    data['signup_date'] = pd.to_datetime(data['signup_date'])

    # Extract year, month, and day from date columns
    data['last_login_year'] = data['last_login'].dt.year
    data['last_login_month'] = data['last_login'].dt.month
    data['last_login_day'] = data['last_login'].dt.day
    data['signup_year'] = data['signup_date'].dt.year
    data['signup_month'] = data['signup_date'].dt.month
    data['signup_day'] = data['signup_date'].dt.day

    # Drop the original date columns
    data = data.drop(columns=['last_login', 'signup_date'])

    # Define columns to be scaled, excluding 'customer_id'
    numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
    numerical_cols = [col for col in numerical_cols if col != 'customer_id']

    # Create a temporary DataFrame for scaling
    temp_data = data[numerical_cols].copy()

    scaler = StandardScaler() # Initialize the StandardScaler
    temp_data = pd.DataFrame(scaler.fit_transform(temp_data), columns=numerical_cols) # Scale numerical columns

    # Encode categorical columns
    encoder = LabelEncoder() # Initialize the LabelEncoder
    categorical_cols = data.select_dtypes(include=['object']).columns.tolist() # Get list of categorical columns
    for col in categorical_cols:
        data[col] = encoder.fit_transform(data[col]) # Encode categorical columns

    # Rejoin customer_id and scaled numerical columns
    data = data.drop(columns=numerical_cols) # Drop original numerical columns
    data = pd.concat([data, temp_data], axis=1) # Concatenate scaled numerical columns back
    data['customer_id'] = customer_id # Reassign customer_id

    return data

def load_and_preprocess_data(postgres_username, postgres_password, postgres_host, postgres_port, postgres_database, cassandra_host, cassandra_port, cassandra_keyspace):

    # Load data from PostgreSQL
    postgres_engine = db_utils.connect_postgresql(postgres_username, postgres_password, postgres_host, postgres_port, postgres_database)
    data_postgres = pd.read_sql_table('customer_data', postgres_engine) # Load PostgreSQL data

    # Load data from Cassandra
    cassandra_session = db_utils.connect_cassandra(cassandra_host, cassandra_port, cassandra_keyspace)
    query = "SELECT * FROM customer_data"
    rows = cassandra_session.execute(query)
    data_cassandra = pd.DataFrame(list(rows)) # Load Cassandra data

    # Preprocess data
    data_postgres_processed = preprocess_postgres_data(data_postgres) # Preprocess PostgreSQL data
    data_cassandra_processed = preprocess_cassandra_data(data_cassandra) # Preprocess Cassandra data

    return data_postgres_processed, data_cassandra_processed