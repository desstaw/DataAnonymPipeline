import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score 


def preprocess_data(data):
    # Preprocessing function
    # Oldpeak to int
    df['oldpeak'] = df['oldpeak'].astype(int) 
    # Categorical to object
    df['sex'] = df['sex'].astype(float) 
    df['cp'] = df['cp'].astype(object) 
    df['fbs'] = df['fbs'].astype(float) 
    df['restecg'] = df['restecg'].astype(object) 
    df['exang'] = df['exang'].astype(float) 
    df['slope'] = df['slope'].astype(object) 
    df['ca'] = df['ca'].astype(object) 
    df['thal'] = df['thal'].astype(object) 
    df['target'] = df['target'].astype(float)

    # Normalize numerical features
    df_norm = df.copy()
    scaler = MinMaxScaler()
    df_norm[['age', 'trestbps', 'chol', 'thalach', 'oldpeak']] = scaler.fit_transform(df_norm[['age', 'trestbps', 'chol', 'thalach', 'oldpeak']])

    # Use pandas's 'get_dummies()' method for hot-encoding
    df_norm = pd.get_dummies(df_norm, columns = ['cp', 'restecg', 'slope', 'ca', 'thal'])
    df_preprocessed = df_norm

    return df_preprocessed 


def classify(data, test_size=0.2, random_state=42):

    # Split the data into features and target
    X = data.iloc[:, data.columns != 'target'].values 
    y = data['target'].values

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # knn classification
    classifier= KNeighborsClassifier(n_neighbors=6)  
    classifier.fit(X_train, y_train)
    y_pred= classifier.predict(X_test) 

    # Record accuracy, recall and f1 score
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test,y_pred)
    f1 = f1_score(y_test,y_pred)

    return accuracy, recall, f1

