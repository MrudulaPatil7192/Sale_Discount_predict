import os
import pickle
import numpy as np
from flask import Flask, render_template_string, request, jsonify

app = Flask(__name__)

# Load pickle model securely
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pkl')
model = None

if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)

# Interactive HTML + CSS Animations + JS Interface
HTML_LAYOUT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SVR Model Predictor</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-gradient: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #311042 100%);
            --glass-bg: rgba(255, 255, 255, 0.05);
            --glass-border: rgba(255, 255, 255, 0.12);
            --accent-glow: #818cf8;
            --accent-hover: #6366f1;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Inter', sans-serif;
        }

        body {
            min-height: 100vh;
            background: var(--bg-gradient);
            color: var(--text-primary);
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 2rem 1rem;
            overflow-x: hidden;
        }

        /* Animated Background Particles */
        .bg-glow {
            position: fixed;
            width: 400px;
            height: 400px;
            background: radial-gradient(circle, rgba(99, 102, 241, 0.25) 0%, rgba(0,0,0,0) 70%);
            top: 20%;
            left: 15%;
            filter: blur(50px);
            animation: float 8s ease-in-out infinite alternate;
            z-index: 0;
        }

        @keyframes float {
            0% { transform: translate(0, 0) scale(1); }
            100% { transform: translate(50px, 30px) scale(1.2); }
        }

        .container {
            position: relative;
            z-index: 1;
            width: 100%;
            max-width: 850px;
            background: var(--glass-bg);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid var(--glass-border);
            border-radius: 24px;
            padding: 2.5rem;
            box-shadow: 0 20px 50px rgba(0, 0, 0, 0.4);
            animation: fadeIn 0.8s cubic-bezier(0.16, 1, 0.3, 1);
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        header {
            text-align: center;
            margin-bottom: 2rem;
        }

        header h1 {
            font-size: 2.2rem;
            font-weight: 700;
            background: linear-gradient(90deg, #a5b4fc, #e0e7ff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }

        header p {
            color: var(--text-secondary);
            font-size: 0.95rem;
        }

        .form-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 1.2rem;
        }

        .input-group {
            display: flex;
            flex-direction: column;
            gap: 0.4rem;
        }

        .input-group label {
            font-size: 0.85rem;
            font-weight: 600;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .input-group input, .input-group select {
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid var(--glass-border);
            border-radius: 12px;
            padding: 0.75rem 1rem;
            color: #fff;
            font-size: 0.95rem;
            outline: none;
            transition: all 0.3s ease;
        }

        .input-group input:focus, .input-group select:focus {
            border-color: var(--accent-glow);
            box-shadow: 0 0 15px rgba(129, 140, 248, 0.3);
        }

        .btn-submit {
            grid-column: 1 / -1;
            margin-top: 1rem;
            background: linear-gradient(90deg, #6366f1, #4f46e5);
            color: #fff;
            border: none;
            padding: 1rem;
            border-radius: 12px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 4px 20px rgba(99, 102, 241, 0.4);
        }

        .btn-submit:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 25px rgba(99, 102, 241, 0.6);
        }

        .btn-submit:active {
            transform: translateY(0);
        }

        /* Result Section */
        #result-card {
            display: none;
            margin-top: 2rem;
            padding: 1.5rem;
            background: rgba(99, 102, 241, 0.1);
            border: 1px solid rgba(129, 140, 248, 0.3);
            border-radius: 16px;
            text-align: center;
            animation: pulseIn 0.5s ease;
        }

        @keyframes pulseIn {
            0% { transform: scale(0.95); opacity: 0; }
            100% { transform: scale(1); opacity: 1; }
        }

        #result-card h3 {
            color: var(--text-secondary);
            font-size: 0.9rem;
            text-transform: uppercase;
        }

        #result-card .prediction-value {
            font-size: 2.5rem;
            font-weight: 700;
            color: #a5b4fc;
            margin-top: 0.3rem;
        }
    </style>
</head>
<body>
    <div class="bg-glow"></div>
    <div class="container">
        <header>
            <h1>SVR Prediction Engine</h1>
            <p>Enter the feature metrics to calculate real-time predictions</p>
        </header>

        <form id="predict-form" class="form-grid">
            <div class="input-group">
                <label>Customer ID</label>
                <input type="number" name="customer_id" value="101" required>
            </div>
            
            <div class="input-group">
                <label>Product Category</label>
                <select name="product_category">
                    <option value="0">Electronics</option>
                    <option value="1">Clothing</option>
                    <option value="2">Home & Kitchen</option>
                    <option value="3">Books</option>
                </select>
            </div>

            <div class="input-group">
                <label>Region</label>
                <select name="region">
                    <option value="0">North</option>
                    <option value="1">South</option>
                    <option value="2">East</option>
                    <option value="3">West</option>
                </select>
            </div>

            <div class="input-group">
                <label>Quantity</label>
                <input type="number" step="any" name="quantity" value="5" required>
            </div>

            <div class="input-group">
                <label>Unit Price ($)</label>
                <input type="number" step="any" name="unit_price" value="49.99" required>
            </div>

            <div class="input-group">
                <label>Payment Method</label>
                <select name="payment_method">
                    <option value="0">Credit Card</option>
                    <option value="1">PayPal</option>
                    <option value="2">Bank Transfer</option>
                </select>
            </div>

            <div class="input-group">
                <label>Delivery Days</label>
                <input type="number" step="any" name="delivery_days" value="3" required>
            </div>

            <div class="input-group">
                <label>Customer Rating</label>
                <input type="number" step="0.1" name="customer_rating" value="4.5" min="1" max="5" required>
            </div>

            <div class="input-group">
                <label>Revenue ($)</label>
                <input type="number" step="any" name="revenue" value="249.95" required>
            </div>

            <button type="submit" class="btn-submit">Generate Prediction</button>
        </form>

        <div id="result-card">
            <h3>Predicted Value</h3>
            <div class="prediction-value" id="prediction-output">--</div>
        </div>
    </div>

    <script>
        document.getElementById('predict-form').addEventListener('submit', async (e) => {
            e.preventDefault();
            const formData = new FormData(e.target);
            const data = Object.fromEntries(formData.entries());

            const res = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });

            const result = await res.json();
            const resultCard = document.getElementById('result-card');
            const output = document.getElementById('prediction-output');

            if (res.ok) {
                output.innerText = Number(result.prediction).toFixed(4);
                resultCard.style.display = 'block';
            } else {
                alert('Error: ' + result.error);
            }
        });
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_LAYOUT)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model pickle file not loaded'}), 500

    try:
        data = request.get_json()
        
        # Parse inputs in order matching feature_names_in_
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
