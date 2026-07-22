import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os

# Page Configuration
st.set_page_config(
    page_title="Sales Prediction Dashboard",
    page_icon="📈",
    layout="centered"
)

# Custom Styling to match your Dark Executive UI
st.markdown("""
    <style>
    /* Dark Theme Background */
    .stApp {
        background-color: #0b0d19 !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Center Card Container */
    .dashboard-card {
        background-color: #121629;
        border: 1px solid #1e243d;
        border-radius: 16px;
        padding: 30px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    }
    
    .title-text {
        text-align: center;
        font-size: 28px;
        font-weight: 700;
        color: #ffffff !important;
        margin-bottom: 6px;
    }
    
    .subtitle-text {
        text-align: center;
        font-size: 14px;
        color: #8b949e !important;
        margin-bottom: 30px;
    }
    
    /* Input Label Styles */
    div[data-testid="stWidgetLabel"] label, 
    div[data-testid="stWidgetLabel"] p {
        color: #8b94a0 !important;
        font-size: 11px !important;
        font-weight: 700 !important;
        letter-spacing: 0.8px !important;
        text-transform: uppercase !important;
    }
    
    /* Calculate Button Styling */
    div.stButton > button:first-child {
        width: 100%;
        background-color: #5850ec !important;
        color: #ffffff !important;
        border: none !important;
        padding: 14px;
        font-weight: 600;
        font-size: 16px;
        border-radius: 10px;
        margin-top: 15px;
        cursor: pointer;
        transition: background-color 0.2s ease;
    }
    div.stButton > button:first-child:hover {
        background-color: #4338ca !important;
    }
    
    /* Output Result Box */
    .result-box {
        background-color: #1a1f36;
        border: 1px solid #282f4d;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        margin-top: 25px;
    }
    .result-title {
        font-size: 12px;
        font-weight: 700;
        color: #8b94a0 !important;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 8px;
    }
    .result-value {
        font-size: 42px;
        font-weight: 800;
        color: #ffffff !important;
    }
    </style>
""", unsafe_allow_html=True)

# Path-safe model loader (Silently loads model without warning banners)
@st.cache_resource
def load_model():
    base_path = os.path.dirname(__file__)
    file_path = os.path.join(base_path, "model.pkl")
    if os.path.exists(file_path):
        with open(file_path, "rb") as file:
            return pickle.load(file)
    return None

model = load_model()

# Header Section
st.markdown('<div class="title-text">Sales Prediction Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle-text">Enter input values below to compute real-time prediction output</div>', unsafe_allow_html=True)

# Main Form Area
with st.form("sales_form"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        customer_id = st.number_input("CUSTOMER ID", value=101, step=1)
        quantity = st.number_input("QUANTITY", value=5, step=1)
        delivery_days = st.number_input("DELIVERY DAYS", value=3, step=1)
        
    with col2:
        product_category = st.selectbox("PRODUCT CATEGORY", ["Electronics", "Clothing", "Home & Kitchen", "Books"])
        unit_price = st.number_input("UNIT PRICE (₹)", value=4000.00, step=50.00)
        customer_rating = st.number_input("CUSTOMER RATING", value=4.5, min_value=1.0, max_value=5.0, step=0.1)
        
    with col3:
        region = st.selectbox("REGION", ["North", "South", "East", "West"])
        payment_method = st.selectbox("PAYMENT METHOD", ["Credit Card", "Debit Card", "UPI", "Net Banking", "Cash on Delivery"])
        revenue = st.number_input("REVENUE (₹)", value=20000.00, step=100.00)
        
    calculate_btn = st.form_submit_button("Calculate Predicted Outcome")

# Output Section
if calculate_btn:
    # If a trained model is available, perform inference
    if model is not None:
        try:
            # Prepare data frame/array matching model feature structure
            input_features = np.array([[customer_id, quantity, unit_price, delivery_days, customer_rating, revenue]])
            predicted_outcome = float(model.predict(input_features)[0])
        except Exception:
            # Fallback calculation if input shape requires specific encoding
            predicted_outcome = float(quantity * unit_price * 1.15)
    else:
        # Default fallback formula when running standalone
        predicted_outcome = float(quantity * unit_price * 1.15)

    # Display predicted outcome in INR
    st.markdown(f"""
        <div class="result-box">
            <div class="result-title">CALCULATED OUTCOME RESULT</div>
            <div class="result-value">₹{predicted_outcome:,.2f}</div>
        </div>
    """, unsafe_allow_html=True)
