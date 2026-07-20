import os
import pickle
import numpy as np
from flask import Flask, render_template_string, request, jsonify

app = Flask(__name__)

# Absolute path resolution
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model.pkl')

model = None
model_load_error = None

# Attempt model loading
if os.path.exists(MODEL_PATH):
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
    except Exception as e:
        model_load_error = f"Failed to deserialize model.pkl: {str(e)}"
else:
    model_load_error = f"model.pkl not found at path: {MODEL_PATH}"

HTML_LAYOUT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Sale Discount Predictor</title>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg: #030712;
            --card-bg: rgba(17, 24, 39, 0.7);
            --border: rgba(255, 255, 255, 0.1);
            --accent: #6366f1;
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

        /* Animated Glowing Orbs */
        .orb {
            position: absolute;
            border-radius: 50%;
            filter: blur(90px);
            z-index: 0;
            animation: float 10s ease-in-out infinite alternate;
        }
        .orb-1 {
            width: 350px;
            height: 350px;
            background: rgba(99, 102, 241, 0.25);
            top: -50px;
            left: -50px;
        }
        .orb-2 {
            width: 300px;
            height: 300px;
            background: rgba(168, 85, 247, 0.2);
            bottom: -50px;
            right: -50px;
            animation-delay: -5s;
        }

        @keyframes float {
            0% { transform: translate(0, 0) scale(1); }
            100% { transform: translate(40px, 30px) scale(1.15); }
        }

        .container {
            position: relative;
            z-index: 1;
            width: 100%;
            max-width: 880px;
            background: var(--card-bg);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border: 1px solid var(--border);
            border-radius: 28px;
            padding: 2.5rem;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
            animation: slideUp 0.6s cubic-bezier(0.16, 1, 0.3, 1);
        }

        @keyframes slideUp {
            from { opacity: 0; transform: translateY(30px); }
            to { opacity: 1; transform: translateY(0); }
        }

        header {
            text-align: center;
            margin-bottom: 2.5rem;
        }

        header h1 {
            font-size: 2.2rem;
            font-weight: 700;
            background: linear-gradient(135deg, #ffffff 0%, #a5b4fc 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: -0.02em;
        }

        header p {
            color: var(--text-sub);
            font-size: 0.95rem;
            margin-top: 0.4rem;
        }

        .status-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.4rem 0.9rem;
            border-radius: 9999px;
            font-size: 0.8rem;
            font-weight: 500;
            margin-bottom: 1rem;
        }
        .status-ok {
            background: rgba(34, 197, 94, 0.1);
            color: #4ade80;
            border: 1px solid rgba(34, 197, 94, 0.2);
        }
        .status-err {
            background: rgba(239, 68, 68, 0.1);
            color: #f87171;
            border: 1px solid rgba(239, 68, 68, 0.2);
        }

        .form-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 1.25rem;
        }

        .input-field {
            display: flex;
            flex-direction: column;
            gap: 0.4rem;
        }

        .input-field label {
            font-size: 0.8rem;
            font-weight: 600;
            color: var(--text-sub);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .input-field input, .input-field select {
            background: rgba(31, 41, 55, 0.6);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 0.8rem 1rem;
            color: #fff;
            font-size: 0.95rem;
            outline: none;
            transition: all 0.2s ease;
        }

        .input-field input:focus, .input-field select:focus {
            border-color: var(--accent);
            box-shadow: 0 0 0 3px var(--accent-glow);
        }

        .btn-predict {
            grid-column: 1 / -1;
            margin-top: 1rem;
            background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
            color: #fff;
            border: none;
            padding: 1rem;
            border-radius: 14px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px var(--accent-glow);
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
        }

        .btn-predict:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px var(--accent-glow);
        }

        /* Prediction Banner Output */
        #result-container {
            display: none;
            margin-top: 2rem;
            padding: 1.5rem;
            background: linear-gradient(135deg, rgba(99, 102, 241, 0.15) 0%, rgba(168, 85, 247, 0.15) 100%);
            border: 1px solid rgba(129, 140, 248, 0.3);
            border-radius: 18px;
            text-align: center;
            animation: fadeIn 0.4s ease-out;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: scale(0.96); }
            to { opacity: 1; transform: scale(1); }
        }

        #result-container h3 {
            font-size: 0.85rem;
            color: #c7d2fe;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }

        .prediction-number {
            font-size: 2.8rem;
            font-weight: 800;
            color: #ffffff;
            margin-top: 0.2rem;
            text-shadow: 0 0 20px var(--accent-glow);
        }
    </style>
</head>
<body>
    <div class="orb orb-1"></div>
    <div class="orb orb-2"></div>

    <div class="container">
        <header>
            {% if model_ready %}
                <div class="status-badge status-ok">● Model Ready</div>
            {% else %}
                <div class="status-badge status-err">● Model Offline</div>
            {% endif %}
            <h1>Sales Prediction AI</h1>
            <p>Enter continuous and categorical feature attributes below</p>
        </header>

        <form id="prediction-form" class="form-grid">
            <div class="input-field">
                <label>Customer ID</label>
                <input type="number" name="customer_id" value="101" required>
            </div>

            <div class="input-field">
                <label>Product Category</label>
                <select name="product_category">
                    <option value="0">Electronics</option>
                    <option value="1">Clothing</option>
                    <option value="2">Home & Kitchen</option>
                    <option value="3">Books</option>
                </select>
            </div>

            <div class="input-field">
                <label>Region</label>
                <select name="region">
                    <option value="0">North</option>
                    <option value="1">South</option>
                    <option value="2">East</option>
                    <option value="3">West</option>
                </select>
            </div>

            <div class="input-field">
                <label>Quantity</label>
                <input type="number" step="any" name="quantity" value="5" required>
            </div>

            <div class="input-field">
                <label>Unit Price ($)</label>
                <input type="number" step="any" name="unit_price" value="49.99" required>
            </div>

            <div class="input-field">
                <label>Payment Method</label>
                <select name="payment_method">
                    <option value="0">Credit Card</option>
                    <option value="1">PayPal</option>
                    <option value="2">Bank Transfer</option>
                </select>
            </div>

            <div class="input-field">
                <label>Delivery Days</label>
                <input type="number" step="any" name="delivery_days" value="3" required>
            </div>

            <div class="input-field">
                <label>Customer Rating</label>
                <input type="number" step="0.1" name="customer_rating" value="4.5" min="1" max="5" required>
            </div>

            <div class="input-field">
                <label>Revenue ($)</label>
                <input type="number" step="any" name="revenue" value="249.95" required>
            </div>

            <button type="submit" class="btn-predict" id="submit-btn">
                <span>Calculate Prediction</span>
            </button>
        </form>

        <div id="result-container">
            <h3>Predicted Target Value</h3>
            <div class="prediction-number" id="output-val">--</div>
        </div>
    </div>

    <script>
        document.getElementById('prediction-form').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const btn = document.getElementById('submit-btn');
            const resultBox = document.getElementById('result-container');
            const outputVal = document.getElementById('output-val');

            btn.disabled = true;
            btn.innerText = 'Processing...';

            const formData = new FormData(e.target);
            const data = Object.fromEntries(formData.entries());

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                });

                const result = await response.json();

                if (response.ok) {
                    outputVal.innerText = Number(result.prediction).toFixed(4);
                    resultBox.style.display = 'block';
                } else {
                    alert('Error: ' + result.error);
                }
            } catch (err) {
                alert('Connection error. Please try again.');
            } finally {
                btn.disabled = false;
                btn.innerText = 'Calculate Prediction';
            }
        });
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_LAYOUT, model_ready=(model is not None))

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': model_load_error or 'Model file not loaded.'}), 500

    try:
        data = request.get_json()
        
        # Order input features according to trained model expectations
        features = [
            float(data['customer_id']),
            float(data['product_category']),
            float(data['region']),
            float(data['quantity']),
            float(data['unit_price']),
            float(data['payment_method']),
            float(data['delivery_days']),
            float(data['customer_rating']),
            float(data['revenue'])
        ]

        features_array = np.array([features])
        prediction = model.predict(features_array)

        return jsonify({'prediction': float(prediction[0])})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
