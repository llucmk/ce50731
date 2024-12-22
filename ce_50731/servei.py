from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# Definim el camí base per als models
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))

# Carreguem els models i preprocessadors
logistic_regression_model = joblib.load(os.path.join(base_path, 'logistic_regression.pkl'))
random_forest_model = joblib.load(os.path.join(base_path, 'random_forest.pkl'))
svm_model = joblib.load(os.path.join(base_path, 'svm.pkl'))
knn_model = joblib.load(os.path.join(base_path, 'knn.pkl'))
dict_vectorizer = joblib.load(os.path.join(base_path, 'dict_vectorizer.pkl'))
scaler = joblib.load(os.path.join(base_path, 'scaler.pkl'))

# Definim el mapeig de les espècies
species_mapping = {0: 'Adelie', 1: 'Chinstrap', 2: 'Gentoo'}

# Funció per preprocessar les dades d'entrada
def preprocess_input(data):
    data_dict = [data]
    transformed_data = dict_vectorizer.transform(data_dict)
    scaled_data = scaler.transform(transformed_data)
    return scaled_data

# Funció per obtenir la predicció del model
def get_prediction(model, data):
    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0].max()
    species = species_mapping[prediction]
    return species, probability

# Ruta per fer prediccions amb el model especificat
@app.route('/predict/<model_name>', methods=['POST'])
def predict(model_name):
    data = request.get_json()
    processed_data = preprocess_input(data)
    
    if model_name == 'logistic_regression':
        species, probability = get_prediction(logistic_regression_model, processed_data)
    elif model_name == 'random_forest':
        species, probability = get_prediction(random_forest_model, processed_data)
    elif model_name == 'svm':
        species, probability = get_prediction(svm_model, processed_data)
    elif model_name == 'knn':
        species, probability = get_prediction(knn_model, processed_data)
    else:
        return jsonify({'error': 'Model not found'}), 404
    
    return jsonify({'species': species, 'probability': probability})

# Executem l'aplicació Flask
if __name__ == '__main__':
    app.run(debug=True)