from flask import Flask, request, jsonify, render_template
import pickle
from spam_detector import preprocess_text
import numpy as np

app = Flask(__name__)

with open('spam_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

@app.route('/')
def home():
    return render_template('front.html')

@app.route('/predict', methods=['POST'])
def predict():
    message = request.form['message']
    preprocessed_message = preprocess_text(message)
    message_tfidf = vectorizer.transform([preprocessed_message])
    prediction = model.predict(message_tfidf)[0]
    confidence = np.max(model.predict_proba(message_tfidf))
    result = "Spam" if prediction == 'spam' else "Not Spam"
    return jsonify({'result': result, 'confidence': float(confidence)})

if __name__ == '__main__':
    app.run(debug=True)
