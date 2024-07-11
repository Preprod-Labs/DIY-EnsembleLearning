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
     
    # Description: This code snippet trains a BaggingClassifier model using the training data stored in a Redis database.
        # Redis: Yes

# CODE - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

    # Dependency: 
        # Environment:     
            # Python 3.11.5
            # Pandas 2.2.2
            # Scikit-learn 1.5.0

import redis                                        # For connecting to Redis database
import pickle                                       # For serializing and deserializing data for storage in Redis
import pandas as pd                                 # For data manipulation
from sklearn.ensemble import RandomForestClassifier # For creating the BaggingClassifier
from sklearn.model_selection import GridSearchCV    # For hyperparameter tuning
from sklearn.metrics import accuracy_score          # For evaluating the model

def read_training_data(r):
    # Load the training features data from Redis and deserialize it
    X_train = pickle.loads(r.get('X_train'))
    
    # Load the training labels data from Redis and deserialize it
    y_train = pickle.loads(r.get('y_train'))
    
    # Convert the features data to a DataFrame
    X_train = pd.DataFrame(X_train)
    
    # Rename the columns to ensure they are strings (necessary for compatibility)
    X_train = X_train.rename(str, axis="columns")
    
    # Convert the labels data to a Series
    y_train = pd.Series(y_train)
    
    # Return the features and labels
    return X_train, y_train

def perform_hyperparameter_tuning(X_train, y_train):
    # Define the grid of hyperparameters to search over
    param_grid = {
        'n_estimators': [50, 100, 150],  # Number of trees in the forest
        'max_depth': [None, 10, 20, 30],  # Maximum depth of the tree
        'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
        'min_samples_leaf': [1, 2, 4],  # Minimum number of samples required to be at a leaf node
        'bootstrap': [True, False]  # Whether bootstrap samples are used when building trees
    }
    
    # Initialize a RandomForestClassifier
    model = RandomForestClassifier(random_state=42)
    
    # Initialize a GridSearchCV to search for the best hyperparameters
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, 
                               cv=3, n_jobs=-1, verbose=2, scoring='accuracy')
    
    # Fit the GridSearchCV to the training data
    grid_search.fit(X_train, y_train)
    
    # Retrieve the best model with the optimal hyperparameters
    best_model = grid_search.best_estimator_
    
    # Print the best hyperparameters found
    print("Best hyperparameters:", grid_search.best_params_)
    
    # Return the best model
    return best_model, grid_search.best_params_

def evaluate_model(model, X_train, y_train):
    try:
        # Predict the training labels using the trained model
        y_pred_train = model.predict(X_train)
        
        # Calculate the accuracy of the model on the training data
        train_accuracy = accuracy_score(y_train, y_pred_train)
        
        # Return the training accuracy
        return train_accuracy
    
    except Exception as e:
        # Print an error message if there is an issue during evaluation
        print("Error during model evaluation:", e)

def save_model(model, model_path):
    # Open a file in write-binary mode to save the model
    with open(model_path, "wb") as f:
        # Serialize the model and save it to the file
        pickle.dump(model, f)

def train_model(redis_host, redis_port, redis_db, model_path):
    
    # Connect to the Redis database
    r = redis.Redis(host=redis_host, port=redis_port, db=redis_db)
    
    # Load the training data from Redis
    X_train, y_train = read_training_data(r)
    
    # Perform hyperparameter tuning to find the best model
    best_model, best_params = perform_hyperparameter_tuning(X_train, y_train)
    
    # Fit the best model to the training data
    best_model.fit(X_train, y_train)
    
    # Print a message indicating the model has been fitted successfully
    print("Best model fitted successfully.")
    
    # Evaluate the best model on the training data
    train_accuracy = evaluate_model(best_model, X_train, y_train)
    
    # Save the best model to a file
    save_model(best_model, model_path)
    
    # Print a message indicating the model training is completed
    print('Model training completed successfully!')
    
    # Return the training accuracy
    return train_accuracy, best_params

# If this script is run as the main program, execute the train_model function
if __name__ == "__main__":
    train_model('localhost', 6379, 1, "bagging_model.pkl")