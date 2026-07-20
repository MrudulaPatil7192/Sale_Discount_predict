import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template_string, request, jsonify

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model.pkl')

model = None
scaler = None
model_status = "Offline"
status_message = ""

FEATURE_COLUMNS = [
    'customer_id',
    'product_category',
    'region',
    'quantity',
    'unit_price',
    'payment_method',
    'delivery_days',
    'customer_rating',
    'revenue'
]

def train_fallback_model():
    """Trains a fallback SVR model if model.pkl is missing or unpickleable."""
    from sklearn.svm import SVR
    X_dummy = pd.DataFrame(np.random.rand(100, 9) * 100, columns=FEATURE_COLUMNS)
    y_dummy = np.random.rand(100) * 500
    svr = SVR(kernel='rbf')
    svr.fit(X_dummy, y_dummy)
    return svr, None

# Safely unpickle model and check for dictionary structure
if os.path.exists(MODEL_PATH):
    try:
        with open(MODEL_PATH, 'rb') as f:
            obj = pickle.load(f)
        
        # Check if pickled object is a dictionary containing model/scaler
        if isinstance(obj, dict):
            model = obj.get('model') or obj.get('svr') or obj.get('regressor') or list(obj.values())[0]
            scaler = obj.get('scaler') or obj.get('std_scaler')
        else:
            model = obj
            
        model_status = "Online"
        status_message = "Custom ML Model loaded successfully."
    except Exception as e:
        model, scaler = train_fallback_model()
        model_status = "Fallback Mode"
        status_message = f"Pickle load error: {str(e)}. Running on Fallback SVR Engine."
else:
    model, scaler = train_fallback_model()
    model_status = "Fallback Mode"
    status_message = "model.pkl not found in root directory. Running on Fallback SVR Engine."

HTML_LAYOUT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Interactive Sales Predictor</title>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg: #030712;
            --card-bg: rgba(17, 24, 39, 0.85);
            --border: rgba(255, 255, 255, 0.12);
            --accent: #6366f1;
            --accent-hover: #4f46e5;
            --accent-glow: rgba(99, 102, 241, 0.4);
            --text-main: #f9fafb;
            --text-sub: #9ca3af;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Plus Jakarta Sans', sans-serif;
        }

        body {
            min-height: 100vh;
            background-color: var(--bg);
            color: var(--text-main);
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 2rem 1rem;
            position: relative;
            overflow-x: hidden;
        }

        /* Ambient Animated Glows */
        .glow-orb {
            position: absolute;
            border-radius: 50%;
            filter: blur(100px);
            z-index: 0;
            animation: pulseGlow 8s ease-in-out infinite alternate;
        }
        .orb-1 {
            width: 420px;
            height: 420px;
            background: rgba(99, 102, 241, 0.22);
            top: -100px;
            left: -100px;
        }
        .orb-2 {
            width: 380px;
            height: 380px;
            background: rgba(168, 85, 247, 0.18);
            bottom: -100px;
            right: -100px;
            animation-delay: -4s;
        }

        @keyframes pulseGlow {
            0% { transform: scale(1) translate(0, 0); }
            100% { transform: scale(1.15) translate(30px, 20px); }
        }

        .container {
            position: relative;
            z-index: 1;
            width: 100%;
            max-width: 900px;
            background: var(--card-bg);
            backdrop-filter: blur(24px);
            -webkit-backdrop-filter: blur(24px);
            border: 1px solid var(--border);
            border-radius: 28px;
            padding: 2.5rem;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.7);
        }

        header {
            text-align: center;
            margin-bottom: 2rem;
        }

        .status-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.35rem 0.9rem;
            border-radius: 9999px;
            font-size: 0.8rem;
            font-weight: 600;
            margin-bottom: 0.8rem;
        }
        .status-online {
            background: rgba(34, 197, 94, 0.12);
            color: #4ade80;
            border: 1px solid rgba(34, 197, 94, 0.3);
        }
        .status-fallback {
            background: rgba(234, 179, 8, 0.12);
            color: #facc15;
            border: 1px solid rgba(234, 179, 8, 0.3);
        }

        header h1 {
            font-size: 2.2rem;
            font-weight: 800;
            background: linear-gradient(135deg, #ffffff 0%, #c7d2fe 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        header p {
            color: var(--text-sub);
            font-size: 0.95rem;
            margin-top: 0.4rem;
        }

        .info-banner {
            margin-bottom: 1.5rem;
            padding: 0.85rem 1rem;
            border-radius: 12px;
            font-size: 0.85rem;
            text-align: center;
        }
        .info-online {
            background: rgba(34, 197, 94, 0.08);
            border: 1px solid rgba(34, 197, 94, 0.2);
            color: #86efac;
        }
        .info-fallback {
            background: rgba(234, 179, 8, 0.08);
            border: 1px solid rgba(234, 179, 8, 0.2);
            color: #fde047;
        }

        .toolbar {
            display: flex;
            justify-content: flex-end;
            gap: 0.75rem;
            margin-bottom: 1rem;
        }

        .btn-secondary {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--border);
            color: var(--text-sub);
            padding: 0.4rem 0.8rem;
            border-radius: 8px;
            font-size: 0.8rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s ease;
        }

        .btn-secondary:hover {
            background: rgba(255, 255, 255, 0.12);
            color: #fff;
        }

        .form-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 1.2rem;
        }

        .input-group {
            display: flex;
            flex-direction: column;
            gap: 0.4rem;
        }

        .input-group label {
            font-size: 0.78rem;
            font-weight: 600;
            color: var(--text-sub);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .input-group input, .input-group select {
            background: rgba(15, 23, 42, 0.7);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 0.75rem 1rem;
            color: #fff;
            font-size: 0.95rem;
            outline: none;
            transition: all 0.25s ease;
        }

        .input-group input:focus, .input-group select:focus {
            border-color: var(--accent);
            box-shadow: 0 0 0 3px var(--accent-glow);
        }

        .btn-submit {
            grid-column: 1 / -1;
            margin-top: 1rem;
            background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
            color: #fff;
            border: none;
            padding: 1rem;
            border-radius: 14px;
            font-size: 1rem;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 4px 20px var(--accent-glow);
        }

        .btn-submit:hover:not(:disabled) {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px var(--accent-glow);
        }

        .btn-submit:disabled {
            opacity: 0.6;
            cursor: not-allowed;
        }

        /* Result Panel */
        #result-card {
            display: none;
            margin-top: 2rem;
            padding: 1.8rem;
            background: linear-gradient(135deg, rgba(99, 102, 241, 0.15) 0%, rgba(168, 85, 247, 0.15) 100%);
            border: 1px solid rgba(129, 140, 248, 0.35);
            border-radius: 20px;
            text-align: center;
            animation: fadeIn 0.4s ease-out;
        }

        #error-card {
            display: none;
            margin-top: 1.5rem;
            padding: 1rem 1.2rem;
            background: rgba(239, 68, 68, 0.12);
            border: 1px solid rgba(239, 68, 68, 0.3);
            border-radius: 14px;
            color: #f87171;
            font-size: 0.88rem;
            text-align: left;
            animation: fadeIn 0.4s ease-out;
            white-space: pre-wrap;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: scale(0.96); }
            to { opacity: 1; transform: scale(1); }
        }

        #result-card h3 {
            font-size: 0.85rem;
            color: #c7d2fe;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }

        .prediction-value {
            font-size: 3rem;
            font-weight: 800;
            color: #ffffff;
            margin-top: 0.3rem;
            text-shadow: 0 0 25px var(--accent-glow);
        }
    </style>
</head>
<body>
    <div class="glow-orb orb-1"></div>
    <div class="glow-orb orb-2"></div>

    <div class="container">
        <header>
            {% if model_status == "Online" %}
                <div class="status-badge status-online">● Model Online</div>
            {% else %}
                <div class="status-badge status-fallback">● {{ model_status }}</div>
            {% endif %}
            <h1>Sales Prediction Dashboard</h1>
            <p>Interactive ML Interface for Real-Time Target Estimation</p>
        </header>

        <div class="info-banner {% if model_status == 'Online' %}info-online{% else %}info-fallback{% endif %}">
            {{ status_message }}
        </div>

        <div class="toolbar">
            <button type="button" class="btn-secondary" onclick="loadPreset(1)">Load Sample A</button>
            <button type="button" class="btn-secondary" onclick="loadPreset(2)">Load Sample B</button>
        </div>

        <form id="predict-form" class="form-grid">
            <div class="input-group">
                <label>Customer ID</label>
                <input type="number" id="customer_id" name="customer_id" value="101" required>
            </div>

            <div class="input-group">
                <label>Product Category</label>
                <select id="product_category" name="product_category">
                    <option value="0">Electronics</option>
                    <option value="1">Clothing</option>
                    <option value="2">Home & Kitchen</option>
                    <option value="3">Books</option>
                </select>
            </div>

            <div class="input-group">
                <label>Region</label>
                <select id="region" name="region">
                    <option value="0">North</option>
                    <option value="1">South</option>
                    <option value="2">East</option>
                    <option value="3">West</option>
                </select>
            </div>

            <div class="input-group">
                <label>Quantity</label>
                <input type="number" step="any" id="quantity" name="quantity" value="5" required>
            </div>

            <div class="input-group">
                <label>Unit Price ($)</label>
                <input type="number" step="any" id="unit_price" name="unit_price" value="49.99" required>
            </div>

            <div class="input-group">
                <label>Payment Method</label>
                <select id="payment_method" name="payment_method">
                    <option value="0">Credit Card</option>
                    <option value="1">PayPal</option>
                    <option value="2">Bank Transfer</option>
                </select>
            </div>

            <div class="input-group">
                <label>Delivery Days</label>
                <input type="number" step="any" id="delivery_days" name="delivery_days" value="3" required>
            </div>

            <div class="input-group">
                <label>Customer Rating</label>
                <input type="number" step="0.1" id="customer_rating" name="customer_rating" value="4.5" min="1" max="5" required>
            </div>

            <div class="input-group">
                <label>Revenue ($)</label>
                <input type="number" step="any" id="revenue" name="revenue" value="249.95" required>
            </div>

            <button type="submit" class="btn-submit" id="btn-submit">
                Calculate Predicted Outcome
            </button>
        </form>

        <div id="error-card"></div>

        <div id="result-card">
            <h3>Predicted Target Outcome</h3>
            <div class="prediction-value" id="prediction-output">--</div>
        </div>
    </div>

    <script>
        function loadPreset(type) {
            if (type === 1) {
                document.getElementById('customer_id').value = 105;
                document.getElementById('product_category').value = 0;
                document.getElementById('region').value = 1;
                document.getElementById('quantity').value = 12;
                document.getElementById('unit_price').value = 199.99;
                document.getElementById('payment_method').value = 0;
                document.getElementById('delivery_days').value = 2;
                document.getElementById('customer_rating').value = 4.8;
                document.getElementById('revenue').value = 2399.88;
            } else {
                document.getElementById('customer_id').value = 202;
                document.getElementById('product_category').value = 2;
                document.getElementById('region').value = 3;
                document.getElementById('quantity').value = 2;
                document.getElementById('unit_price').value = 29.50;
                document.getElementById('payment_method').value = 1;
                document.getElementById('delivery_days').value = 5;
                document.getElementById('customer_rating').value = 3.9;
                document.getElementById('revenue').value = 59.00;
            }
        }

        document.getElementById('predict-form').addEventListener('submit', async (e) => {
            e.preventDefault();
            const btn = document.getElementById('btn-submit');
            const resultCard = document.getElementById('result-card');
            const errorCard = document.getElementById('error-card');
            const output = document.getElementById('prediction-output');

            btn.disabled = true;
            btn.innerText = 'Processing Model...';
            resultCard.style.display = 'none';
            errorCard.style.display = 'none';

            const formData = new FormData(e.target);
            const data = Object.fromEntries(formData.entries());

            try {
                const res = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                });

                const result = await res.json();

                if (res.ok && result.prediction !== undefined) {
                    output.innerText = Number(result.prediction).toFixed(4);
                    resultCard.style.display = 'block';
                } else {
                    errorCard.innerText = 'Backend Prediction Error:\n' + (result.error || 'Unknown response error.');
                    errorCard.style.display = 'block';
                }
            } catch (err) {
                errorCard.innerText = 'Network Error: Unable to reach the backend endpoint.';
                errorCard.style.display = 'block';
            } finally {
                btn.disabled = false;
                btn.innerText = 'Calculate Predicted Outcome';
            }
        });
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(
        HTML_LAYOUT, 
        model_status=model_status, 
        status_message=status_message
    )

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Build dictionary matching feature names
        input_data = {
            'customer_id': [float(data['customer_id'])],
            'product_category': [float(data['product_category'])],
            'region': [float(data['region'])],
            'quantity': [float(data['quantity'])],
            'unit_price': [float(data['unit_price'])],
            'payment_method': [float(data['payment_method'])],
            'delivery_days': [float(data['delivery_days'])],
            'customer_rating': [float(data['customer_rating'])],
            'revenue': [float(data['revenue'])]
        }

        df_input = pd.DataFrame(input_data)

        # Apply scaling if standard scaler was preserved in pickle
        if scaler is not None:
            processed_input = scaler.transform(df_input)
        else:
            processed_input = df_input

        # Predict with DataFrame or NumPy array depending on estimator requirements
        try:
            prediction = model.predict(processed_input)
        except Exception:
            prediction = model.predict(df_input.to_numpy())

        # Extract numeric prediction output
        val = float(prediction[0]) if hasattr(prediction, '__getitem__') else float(prediction)
        return jsonify({'prediction': val})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
