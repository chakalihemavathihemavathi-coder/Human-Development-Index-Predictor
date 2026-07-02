from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

with open("model.pkl", "rb") as file:
    model = pickle.load(file)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    life = float(request.form["life"])
    education = float(request.form["education"])
    income = float(request.form["income"])

    prediction = model.predict(np.array([[life, education, income]]))

    return render_template(
        "index.html",
        prediction_text=f"Predicted HDI: {prediction[0]:.3f}"
    )

if __name__ == "__main__":
    app.run(debug=True)
