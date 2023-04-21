import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score 
from v1_wrapper import preprocess_data, classify

# Read the dataset
url = "https://raw.githubusercontent.com/desstaw/DataScience_Seminar_SS23/main/heart.csv?token=GHSAT0AAAAAACBUW6ZIFPU4S2XITJWNCPOMZCBQ34Q"
df = pd.read_csv(url)


# Preprocess the data using the preprocess_data() function
df_preprocessed = preprocess_data(df)

# Perform classification and get the evaluation metrics' scores before anonymization
accuracy, recall, f1 = classify(df_preprocessed, test_size=0.2, random_state=42)

# Print the score of each metric
print("Accuracy before Anonymization:", accuracy)
print("Recall before Anonymization:", recall)
print("F1: before Anonymization", f1)

# Define a privacy perserving technique
def anoymize(data, arg1, arg2)




    return anonymized_data, privacy_score


# Call anonymize function and save the returned values
anonymized_data, privacy_score = anoymize(preprocessed_data, arg1, arg2)


print("Privacy Score:", privacy_score)




accuracy, recall, f1 = classify(anonymized_data, test_size=0.2, random_state=42)


# Print the score of each metric
print("Accuracy after Anonymization:", accuracy)
print("Recall after Anonymization:", recall)
print("F1: after Anonymization", f1)