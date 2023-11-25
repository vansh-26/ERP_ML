from flask import Flask, request, jsonify
import pandas as pd
import regresionmodel

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})
 
    if file and file.filename.endswith('.csv'):
        df = pd.read_csv(file)

        # Make sure to convert the predictions to a list
        predictions = regresionmodel.read_and_predict(df).tolist()

        return jsonify({'predictions': predictions})

    else:
        return jsonify({'error': 'Invalid file format'})

if __name__ == '__main__':
    app.run(debug=True)
