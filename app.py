from flask import Flask, request, jsonify
import joblib
import spacy
import nltk
from nltk.tokenize import word_tokenize

app = Flask(__name__)

# Load the saved model
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('vectorizer.joblib')
nlp = spacy.load("en_core_web_sm")
def preprocess_text(text):

    # Convert to lowercase
    text = text.lower()
    tokens = word_tokenize(text)
    trimmed_text = " ".join(tokens)
    doc = nlp(trimmed_text)

    # Remove punctuation
    text = " ".join([word for word in[token.text for token in doc if not token.is_stop and not token.is_punct]])

    return text

# create a route that manages user request and does sentiment prediction
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = preprocess_text(data['text'])
    vectorized_text = vectorizer.transform([text])
    prediction = model.predict(vectorized_text).item()
    return jsonify({'sentiment': prediction})


if __name__ == '__main__':
    app.run()