import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import pickle

def train_and_save_model(data_path):
    df = pd.read_csv(data_path)

    label_encoder = LabelEncoder()
    df['Attendance'] = label_encoder.fit_transform(df['Attendance'])
    df['Grade'] = label_encoder.fit_transform(df['Grade'])
    df['Scholarship'] = label_encoder.fit_transform(df['Scholarship'])

    X = df[['Attendance', 'Weekly_Study_Hours', 'Grade']]
    y = df['Scholarship']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(random_state=42)

    model.fit(X_train, y_train)

    with open('logistic_regression_model.pkl', 'wb') as file:
        pickle.dump(model, file)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

train_and_save_model('/content/StudentsExtraCurricular.csv')
