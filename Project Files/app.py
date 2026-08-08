from flask import Flask, render_template, request
import pickle
import numpy as np

# Create Flask application
app = Flask(__name__)

# Load the trained model
with open("hdi_model.pkl", "rb") as file:
    model = pickle.load(file)


# Home page
@app.route("/")
def home():
    return render_template("index.html")


# Prediction page
@app.route("/predict", methods=["POST"])
def predict():

    # Get values entered by the user
    life_expectancy = float(request.form["life_expectancy"])
    expected_schooling = float(request.form["expected_schooling"])
    mean_schooling = float(request.form["mean_schooling"])
    gni = float(request.form["gni"])

    # Create input array
    input_data = np.array([[
        life_expectancy,
        expected_schooling,
        mean_schooling,
        gni
    ]])

    # Predict HDI
    prediction = model.predict(input_data)[0]

    # Display result
    return render_template(
        "index.html",
        prediction=round(prediction, 3)
    )


# Run Flask application
if __name__ == "__main__":
    app.run(debug=True)