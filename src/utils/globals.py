from sklearn.model_selection import train_test_split
import pandas as pd


data = pd.read_csv("data/heart_failure_clinical_records_dataset.csv")

X = data.drop('DEATH_EVENT', axis=1)
y = data["DEATH_EVENT"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)
