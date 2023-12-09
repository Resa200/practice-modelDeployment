from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the saved model
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('vectorizer.joblib')


# create a route that manages user request and does sentiment prediction
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data['text']
    # text = preprocess_text(data['text'])
    vectorized_text = vectorizer.transform([text])
    prediction = model.predict(vectorized_text).item()
    return jsonify({'sentiment': prediction})


if __name__ == '__main__':
    app.run()