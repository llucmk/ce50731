from flask import Flask, request, jsonify
import joblib
import os
import pandas as pd
import numpy as np

app = Flask(__name__)

# Definim el camí base per als models
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))

# Carreguem els models
logistic_regression_model = joblib.load(os.path.join(base_path, 'logistic_regression.pkl'))
decision_tree_model = joblib.load(os.path.join(base_path, 'decision_tree.pkl'))
svm_model = joblib.load(os.path.join(base_path, 'svm.pkl'))
knn_model = joblib.load(os.path.join(base_path, 'knn.pkl'))

# Carreguem el vectoritzador i l'escalador utilitzats a classificacio_propia.ipynb
vectorizer = joblib.load(os.path.join(base_path, 'vectorizer.pkl'))
scaler = joblib.load(os.path.join(base_path, 'scaler.pkl'))

# Definim el mapeig de les espècies
species_mapping = {0: 'Adelie', 1: 'Chinstrap', 2: 'Gentoo'}

# Funció per preprocessar les dades d'entrada
def preprocess_input(data):
    df = pd.DataFrame([data])
    categorical_features = df.select_dtypes(include=["object"]).columns
    numerical_features = df.select_dtypes(exclude=["object"]).columns
    
    # Codificació One-Hot
    data_dict = df[categorical_features].to_dict(orient='records')
    data_categorical = vectorizer.transform(data_dict)
    
    # Escalar les característiques numèriques
    data_numerical = scaler.transform(df[numerical_features])
    
    # Combinar característiques numèriques i categòriques
    data_prepared = np.hstack((data_numerical, data_categorical))
    
    return data_prepared

# Funció per obtenir la predicció del model
def get_prediction(model, data):
    prediction = model.predict(data)[0]
    species = species_mapping[prediction]
    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(data)[0].max()
    else:
        probability = "N/A"
    return species, probability

# Ruta per fer prediccions amb el model especificat
@app.route('/predict/<model_name>', methods=['POST'])
def predict(model_name):
    data = request.get_json()
    processed_data = preprocess_input(data)
    
    if model_name == 'logistic_regression':
        species, probability = get_prediction(logistic_regression_model, processed_data)
    elif model_name == 'decision_tree':
        species, probability = get_prediction(decision_tree_model, processed_data)
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