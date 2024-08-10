from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Carregar o modelo salvo
model = joblib.load('random_forest_model.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obter dados da solicitação
        data = request.json
        # Converter dados para o formato necessário
        features = np.array(data['features']).reshape(1, -1)
        # Fazer previsão
        prediction = model.predict(features)
        # Retornar a previsão como resposta JSON
        return jsonify({'prediction': int(prediction[0])})
    except Exception as e:
        # Retornar erro em caso de exceção
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
