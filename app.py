from flask import Flask, request, render_template_string
import pickle
import numpy as np

app = Flask(__name__)

# Load Model
with open("svm_model.pkl", "rb") as file:
    model = pickle.load(file)

HTML = """

<!DOCTYPE html>
<html>
<head>

<title>SVR Prediction App</title>

<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap" rel="stylesheet">

<style>

*{
margin:0;
padding:0;
box-sizing:border-box;
font-family:'Poppins',sans-serif;
}

body{

background:linear-gradient(-45deg,#0f2027,#203a43,#2c5364,#1c92d2);
background-size:400% 400%;
animation:bg 15s ease infinite;
height:100vh;
display:flex;
justify-content:center;
align-items:center;
overflow:auto;

}

@keyframes bg{

0%{background-position:0% 50%;}
50%{background-position:100% 50%;}
100%{background-position:0% 50%;}

}

.card{

width:700px;
padding:35px;
background:rgba(255,255,255,.12);
border-radius:20px;
backdrop-filter:blur(15px);
box-shadow:0 15px 40px rgba(0,0,0,.35);
animation:fade 1.2s;

}

@keyframes fade{

from{
opacity:0;
transform:translateY(30px);
}

to{
opacity:1;
transform:translateY(0px);
}

}

h1{

text-align:center;
color:white;
margin-bottom:25px;
font-size:34px;

}

.grid{

display:grid;
grid-template-columns:repeat(2,1fr);
gap:15px;

}

input{

padding:13px;
border:none;
outline:none;
border-radius:10px;
font-size:15px;
transition:.3s;

}

input:focus{

transform:scale(1.03);
box-shadow:0 0 15px cyan;

}

button{

margin-top:20px;
width:100%;
padding:14px;
font-size:18px;
border:none;
border-radius:10px;
background:#00e5ff;
color:black;
font-weight:bold;
cursor:pointer;
transition:.3s;

}

button:hover{

background:#00bcd4;
transform:scale(1.03);

}

.result{

margin-top:25px;
padding:18px;
background:rgba(255,255,255,.15);
border-radius:10px;
text-align:center;
font-size:22px;
color:#fff;
animation:zoom .8s;

}

@keyframes zoom{

from{
transform:scale(.5);
opacity:0;
}

to{
transform:scale(1);
opacity:1;
}

}

.footer{

text-align:center;
color:white;
margin-top:15px;
font-size:13px;

}

</style>

</head>

<body>

<div class="card">

<h1>📊 Sales Prediction using SVR</h1>

<form method="POST">

<div class="grid">

<input type="number" step="any" name="customer_id" placeholder="Customer ID" required>

<input type="number" step="any" name="product_category" placeholder="Product Category" required>

<input type="number" step="any" name="region" placeholder="Region" required>

<input type="number" step="any" name="quantity" placeholder="Quantity" required>

<input type="number" step="any" name="unit_price" placeholder="Unit Price" required>

<input type="number" step="any" name="payment_method" placeholder="Payment Method" required>

<input type="number" step="any" name="delivery_days" placeholder="Delivery Days" required>

<input type="number" step="any" name="customer_rating" placeholder="Customer Rating" required>

<input type="number" step="any" name="revenue" placeholder="Revenue" required>

</div>

<button type="submit">
🚀 Predict
</button>

</form>

{% if prediction %}

<div class="result">

Prediction :
<br><br>

<b>{{prediction}}</b>

</div>

{% endif %}

<div class="footer">

Made with ❤️ using Flask & Machine Learning

</div>

</div>

</body>

</html>

"""

@app.route("/", methods=["GET","POST"])

def home():

    prediction=None

    if request.method=="POST":

        values=[

        float(request.form["customer_id"]),
        float(request.form["product_category"]),
        float(request.form["region"]),
        float(request.form["quantity"]),
        float(request.form["unit_price"]),
        float(request.form["payment_method"]),
        float(request.form["delivery_days"]),
        float(request.form["customer_rating"]),
        float(request.form["revenue"])

        ]

        prediction=model.predict([values])[0]

        prediction=round(float(prediction),2)

    return render_template_string(HTML,prediction=prediction)

if __name__=="__main__":

    app.run(host="0.0.0.0",port=5000)
