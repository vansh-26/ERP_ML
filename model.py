import pickle

def function(value):
    with open('sgpa_predict.pkl', 'rb') as file:
        data = pickle.load(file)
    return data.predict([value])[0]

