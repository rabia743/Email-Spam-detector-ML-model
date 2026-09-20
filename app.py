# ============================================================
# SPAM DETECTOR — FLASK BACKEND
# ============================================================
# File: app.py
# Model: Logistic Regression
# Purpose: ML model ko web API ke through available karna
# ============================================================

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os
import sys

# ============================================================
# FLASK APP INITIALIZE
# ============================================================
app = Flask(__name__)
CORS(app)

print("=" * 60)
print("SPAM DETECTOR BACKEND (Logistic Regression)")
print("=" * 60)


# ============================================================
# MODEL + VECTORIZER LOAD KARO
# ============================================================
MODEL_PATH = "logistic_model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"

# Check model file
if not os.path.exists(MODEL_PATH):
    print(f"\n ERROR: '{MODEL_PATH}' nahi mili!")
    print(f"Current folder: {os.getcwd()}")
    print("\n Solution:")
    print("   1. Pehle 'logistic regression.py' chalao")
    print("   2. Usme yeh line add karo: joblib.dump(model, 'logistic_model.pkl')")
    print("   3. Phir 'app.py' chalao")
    print("=" * 60)
    sys.exit(1)

# Check vectorizer file
if not os.path.exists(VECTORIZER_PATH):
    print(f"\n ERROR: '{VECTORIZER_PATH}' nahi mili!")
    print(f" Current folder: {os.getcwd()}")
    print("\n Solution: 'feature eng.py' chalao jo vectorizer save karta hai")
    print("=" * 60)
    sys.exit(1)

# Load model + vectorizer
try:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    print(f"Model loaded: {MODEL_PATH}")
    print(f"Vectorizer loaded: {VECTORIZER_PATH}")
except Exception as e:
    print(f"\n Load error: {e}")
    sys.exit(1)


# ============================================================
# ROUTE 1: HEALTH CHECK
# ============================================================
@app.route('/health', methods=['GET'])
def health():
    """Frontend check karta hai ke backend chal raha hai ya nahi"""
    return jsonify({
        'status': 'ok',
        'model': 'logistic_regression',
        'model_loaded': True,
        'vectorizer_loaded': True,
        'message': 'Backend chal raha hai'
    })


# ============================================================
# ROUTE 2: PREDICT
# ============================================================
@app.route('/predict', methods=['POST'])
def predict():
    """
    Request:  { "text": "your email message" }
    Response: {
        "label": "spam" | "ham",
        "spam_probability": 92.45,
        "ham_probability": 7.55,
        "confidence": 92.45
    }
    """
    try:
        # 1. Data receive karo
        data = request.get_json()

        if not data or 'text' not in data:
            return jsonify({'error': 'Missing "text" field'}), 400

        text = data['text'].strip()

        if not text:
            return jsonify({'error': 'Empty text'}), 400

        # 2. Text ko vectorize karo
        text_vector = vectorizer.transform([text.lower()])

        # 3. Model se prediction lo
        proba = model.predict_proba(text_vector)[0]
        pred = model.predict(text_vector)[0]

        # 4. Percentages nikalo
        spam_prob = float(proba[1]) * 100
        ham_prob = float(proba[0]) * 100
        confidence = max(spam_prob, ham_prob)

        # 5. Response bhejo
        result = {
            'label': 'spam' if pred == 1 else 'ham',
            'spam_probability': round(spam_prob, 2),
            'ham_probability': round(ham_prob, 2),
            'confidence': round(confidence, 2)
        }

        # Terminal mein log
        print(f"\nInput: {text[:60]}{'...' if len(text) > 60 else ''}")
        print(f"Result: {result['label'].upper()} "
              f"(Spam: {result['spam_probability']}%, "
              f"Confidence: {result['confidence']}%)")

        return jsonify(result)

    except Exception as e:
        print(f"\nError: {e}")
        return jsonify({'error': str(e)}), 500


# ============================================================
# ROUTE 3: ROOT
# ============================================================
@app.route('/', methods=['GET'])
def root():
    return jsonify({
        'name': 'Spam Detector API',
        'model': 'Logistic Regression',
        'endpoints': {
            '/health': 'GET — Backend status',
            '/predict': 'POST — Spam prediction'
        }
    })


# ============================================================
# ERROR HANDLERS
# ============================================================
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint nahi mila'}), 404


@app.errorhandler(500)
def server_error(error):
    return jsonify({'error': 'Server error'}), 500


# ============================================================
# MAIN — SERVER CHALAO
# ============================================================
if __name__ == '__main__':
    print("\n" + "=" * 60)
    print(" SERVER START HO RAHA HAI...")
    print("=" * 60)
    print(" URL:     http://127.0.0.1:5000")
    print("Health:  http://127.0.0.1:5000/health")
    print(" Predict: POST http://127.0.0.1:5000/predict")
    print(" Stop:    Ctrl+C")
    print("=" * 60 + "\n")

    app.run(host='0.0.0.0', port=5000, debug=True)