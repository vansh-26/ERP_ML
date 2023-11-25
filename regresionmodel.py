import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def read_and_predict(std_data):
    X = std_data[['Overall_Attendance']]
    y = std_data['Last_Semester_CGPA']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred
