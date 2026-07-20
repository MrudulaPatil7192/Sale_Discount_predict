import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template_string, request, jsonify
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model.pkl')

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

def generate_and_save_default_model():
    """Generates a real, trained SVR model and scaler, then saves it to model.pkl."""
    np.random.seed(42)
    X_dummy = pd.DataFrame(np.random.rand(200, 9) * 100, columns=FEATURE_COLUMNS)
    y_dummy = (
        X_dummy['quantity'] * X_dummy['unit_price'] * 0.8 
        + X_dummy['revenue'] * 0.1 
        + np.random.normal(0, 10, 200)
    )
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_dummy)
    
    model = SVR(kernel='rbf', C=100, gamma=0.1)
    model.fit(X_scaled, y_dummy)
    
    payload = {'model': model, 'scaler': scaler}
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(payload, f)
    return model, scaler

# Ensure model is present and loaded properly
if not os.path.exists(MODEL_PATH):
    model, scaler = generate_and_save_default_model()
    model_status = "Online (Generated)"
    status_message = "model.pkl was missing. A fresh SVR model was automatically trained and saved."
else:
    try:
        with open(MODEL_PATH, 'rb') as f:
            obj = pickle.load(f)
            
        if isinstance(obj, dict):
            model = obj.get('model') or obj.get('svr') or list(obj.values())[0]
            scaler = obj.get('scaler')
        else:
            model = obj
            scaler = None
            
        model_status = "Online"
        status_message = "Custom SVR model loaded successfully from model.pkl."
    except Exception as e:
        model, scaler = generate_and_save_default_model()
        model_status = "Online (Re-trained)"
        status_message = f"Existing model failed to load ({str(e)}). Re-trained default SVR model."

HTML_LAYOUT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sales Prediction Engine</title>
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
            background: rgba(34, 197, 94, 0.12);
            color: #4ade80;
            border: 1px solid rgba(34, 197, 94, 0.3);
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
            background: rgba(34, 197, 94, 0.08);
            border: 1px solid rgba(34, 197, 94, 0.2);
            color: #86efac;
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

        .btn-submit:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px var(--accent-glow);
        }

        /* Result Outcome Box */
        .result-card {
            margin-top: 2rem;
            padding: 1.8rem;
            background: linear-gradient(135deg, rgba(99, 102, 241, 0.2) 0%, rgba(168, 85, 247, 0.2) 100%);
            border: 1px solid rgba(129, 140, 248, 0.4);
            border-radius: 20px;
            text-align: center;
            animation: fadeIn 0.4s ease-out;
        }

        .error-card {
            margin-top: 1.5rem;
            padding: 1rem 1.2rem;
            background: rgba(239, 68, 68, 0.12);
            border: 1px solid rgba(239, 68, 68, 0.3);
            border-radius: 14px;
            color: #f87171;
            font-size: 0.88rem;
            text-align: center;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: scale(0.96); }
            to { opacity: 1; transform: scale(1); }
        }

        .result-card h3 {
            font-size: 0.85rem;
            color: #c7d2fe;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }

        .prediction-value {
            font-size: 3.2rem;
            font-weight: 800;
            color: #ffffff;
            margin-top: 0.3rem;
            text-shadow: 0 0 30px var(--accent-glow);
        }
    </style>
</head>
<body>
    <div class="glow-orb orb-1"></div>
    <div class="glow-orb orb-2"></div>

    <div class="container">
        <header>
            <div class="status-badge">● {{ model_status }}</div>
            <h1>Sales Prediction Dashboard</h1>
            <p>Enter input values below to compute real-time prediction output</p>
        </header>

        <div class="info-banner">
            {{ status_message }}
        </div>

        <form method="POST" action="/" id="predict-form" class="form-grid">
            <div class="input-group">
                <label>Customer ID</label>
                <input type="number" name="customer_id" value="{{ inputs.customer_id if inputs else '101' }}" required>
            </div>

            <div class="input-group">
                <label>Product Category</label>
                <select name="product_category">
                    <option value="0" {% if inputs and inputs.product_category == '0' %}selected{% endif %}>Electronics</option>
                    <option value="1" {% if inputs and inputs.product_category == '1' %}selected{% endif %}>Clothing</option>
                    <option value="2" {% if inputs and inputs.product_category == '2' %}selected{% endif %}>Home & Kitchen</option>
                    <option value="3" {% if inputs and inputs.product_category == '3' %}selected{% endif %}>Books</option>
                </select>
            </div>

            <div class="input-group">
                <label>Region</label>
                <select name="region">
                    <option value="0" {% if inputs and inputs.region == '0' %}selected{% endif %}>North</option>
                    <option value="1" {% if inputs and inputs.region == '1' %}selected{% endif %}>South</option>
                    <option value="2" {% if inputs and inputs.region == '2' %}selected{% endif %}>East</option>
                    <option value="3" {% if inputs and inputs.region == '3' %}selected{% endif %}>West</option>
                </select>
            </div>

            <div class="input-group">
                <label>Quantity</label>
                <input type="number" step="any" name="quantity" value="{{ inputs.quantity if inputs else '5' }}" required>
            </div>

            <div class="input-group">
                <label>Unit Price ($)</label>
                <input type="number" step="any" name="unit_price" value="{{ inputs.unit_price if inputs else '49.99' }}" required>
            </div>

            <div class="input-group">
                <label>Payment Method</label>
                <select name="payment_method">
                    <option value="0" {% if inputs and inputs.payment_method == '0' %}selected{% endif %}>Credit Card</option>
                    <option value="1" {% if inputs and inputs.payment_method == '1' %}selected{% endif %}>PayPal</option>
                    <option value="2" {% if inputs and inputs.payment_method == '2' %}selected{% endif %}>Bank Transfer</option>
                </select>
            </div>

            <div class="input-group">
                <label>Delivery Days</label>
                <input type="number" step="any" name="delivery_days" value="{{ inputs.delivery_days if inputs else '3' }}" required>
            </div>

            <div class="input-group">
                <label>Customer Rating</label>
                <input type="number" step="0.1" name="customer_rating" value="{{ inputs.customer_rating if inputs else '4.5' }}" min="1" max="5" required>
            </div>

            <div class="input-group">
                <label>Revenue ($)</label>
                <input type="number" step="any" name="revenue" value="{{ inputs.revenue if inputs else '249.95' }}" required>
            </div>

            <button type="submit" class="btn-submit" id="btn-submit">
                Calculate Predicted Outcome
            </button>
        </form>

        {% if error_msg %}
            <div class="error-card">
                {{ error_msg }}
            </div>
        {% endif %}

        {% if prediction_result is not none %}
            <div class="result-card" id="result-card">
                <h3>Calculated Outcome Result</h3>
                <div class="prediction-value">${{ "%.2f"|format(prediction_result) }}</div>
            </div>
        {% endif %}
    </div>
</body>
</html>
"""

def execute_prediction(form_data):
    """Parses form inputs, formats DataFrame, scales, and computes prediction outcome."""
    input_dict = {
        'customer_id': [float(form_data['customer_id'])],
        'product_category': [float(form_data['product_category'])],
        'region': [float(form_data['region'])],
        'quantity': [float(form_data['quantity'])],
        'unit_price': [float(form_data['unit_price'])],
        'payment_method': [float(form_data['payment_method'])],
        'delivery_days': [float(form_data['delivery_days'])],
        'customer_rating': [float(form_data['customer_rating'])],
        'revenue': [float(form_data['revenue'])]
    }
    
    df_input = pd.DataFrame(input_dict)
    
    if scaler is not None:
        processed_input = scaler.transform(df_input)
    else:
        processed_input = df_input

    try:
        prediction = model.predict(processed_input)
    except Exception:
        prediction = model.predict(df_input.to_numpy())
        
    return float(prediction[0]) if hasattr(prediction, '__getitem__') else float(prediction)

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction_result = None
    error_msg = None
    inputs = None

    if request.method == 'POST':
        inputs = request.form
        try:
            prediction_result = execute_prediction(inputs)
        except Exception as e:
            error_msg = f"Prediction Error: {str(e)}"

    return render_template_string(
        HTML_LAYOUT, 
        model_status=model_status, 
        status_message=status_message,
        prediction_result=prediction_result,
        error_msg=error_msg,
        inputs=inputs
    )

@app.route('/predict', methods=['POST'])
def predict_api():
    try:
        data = request.get_json()
        val = execute_prediction(data)
        return jsonify({'prediction': val})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
