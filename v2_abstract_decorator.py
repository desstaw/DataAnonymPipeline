from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from functools import wraps
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score 


def anonymize_decorator(classifier):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):

            # Split the original dataset into features and labels
            X = args[0].iloc[:, :-1]
            y = args[0].iloc[:, -1]

            # Split into training and testing 
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

            # Train the classifier on the original training dataset
            classifier.fit(X_train, y_train)

            # Predict labels on the original dataset and calculate the evaluation metrics
            original_y_pred = classifier.predict(X_test)

            original_accuracy = accuracy_score(y_test, original_y_pred)
            original_recall = recall_score(y_test, original_y_pred)
            original_f1 = f1_score(y_test, original_y_pred)

            # Print the evaluation metrics results
            print("Accuracy before privacy preserving technique:", original_accuracy)
            print("Recall before privacy preserving technique:", original_recall)
            print("F1 before privacy preserving technique:", original_f1)
            
            # Apply the privacy preserving technique to the dataset
            anonym_dataset = func(*args, **kwargs)

            # Split the anonymized dataset into features and labels
            anonym_X = anonym_dataset.iloc[:, :-1]
            anonym_y = anonym_dataset.iloc[:, -1]

            # Split the anonymized dataset into training and testing
            anonym_X_train, anonym_X_test, anonym_y_train, anonym_y_test = train_test_split(anonym_X, anonym_y, test_size=0.2)

            # Train the classifier on the anonymized training dataset
            classifier.fit(anonym_X_train, anonym_y_train)

            # Predict labels on the anonymized dataset and calculate the evaluation metrics
            anonym_y_pred = classifier.predict(anonym_X_test)
            
            anonym_accuracy = accuracy_score(anonym_y_test, anonym_y_pred)
            anonym_recall = recall_score(anonym_y_test, anonym_y_pred)
            anonym_f1 = f1_score(anonym_y_test, anonym_y_pred)

            # Print the evaluation metrics results
            print("Accuracy after privacy preserving technique:", anonym_accuracy)
            print("Recall after privacy preserving technique:", anonym_recall)
            print("F1 after privacy preserving technique:", anonym_f1)

            # Return the anonymized dataset
            return anonym_dataset
        return wrapper
    return decorator



# Create an instance of the KNeighborsClassifier (or any other classifier)
knn_classifier = KNeighborsClassifier(n_neighbors=k)

# Privacy preserving function decorator
@anonymize_decorator(knn_classifier)
def anonymize(dataset):
    
    anonym_dataset = ...

    return anonym_dataset


# Read the dataset
url = "github url"
df = pd.read_csv(url)

# Preprocess the data using a predefined preprocess_data() function
df_preprocessed = preprocess_data(df)


# Use the decorated function with the dataset
anonym_dataset = anonymize(df_preprocessed)