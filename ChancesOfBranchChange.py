import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pickle

df = pd.read_csv('BranchChange_Predict.csv')

X = df.drop(['Serial No.', 'Chance of Admit', 'University Rating', 'SOP', 'LOR', 'Research','GRE Score','TOEFL Score'], axis=1)
y = df['Chance of Admit']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()

model.fit(X_train, y_train)

with open('linear_regression_model.pkl', 'wb') as file:
    pickle.dump(model, file)

def predict_chance_of_admit(new_data):
    with open('linear_regression_model.pkl', 'rb') as file:
        loaded_model = pickle.load(file)

    predictions = loaded_model.predict(new_data)

    return predictions
print(predict_chance_of_admit([[7.7]]))