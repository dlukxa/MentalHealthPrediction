from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from flask_cors import CORS #Import CORS extension


# Load dataset and train model
df = pd.read_csv('mental_health_data.csv')
X = df.drop(['Depression'], axis=1)  # Independent features
y = df['Depression']  # Dependent feature

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=17)

# Train logistic regression model
lr_classifier = LogisticRegression(max_iter=1000)
lr_classifier.fit(X_train, y_train)

# Initialize Flask app
app = Flask(__name__)
CORS(app) #Enable CORS for all routes

@app.route('/')
def index():
    return render_template('index.html')

def get_risk_level(probability):
    if probability >= 0.8:
        return "High Risk"
    elif probability >= 0.6:
        return "Medium Risk"
    elif probability >= 0.4:
        return "Low Risk"
    else:
        return "Very Low Risk"

def get_probability_description(probability):
    risk_level = get_risk_level(probability)
    probability = round(probability, 2)
    if risk_level == "High Risk":
        return f"There is a high probability ({probability}) of depression based on this data. It is strongly recommended to seek professional help immediately."
    elif risk_level == "Medium Risk":
        return f"There is a moderate probability ({probability}) of depression.  Consider seeking guidance from a healthcare provider soon."
    elif risk_level == "Low Risk":
        return f"There is a low probability ({probability}) of depression based on this data. However, maintaining healthy habits and monitoring your mental well-being is still important."
    else:  # Very Low Risk
        return f"The probability ({probability}) of depression is currently very low based on this data. Maintaining a healthy lifestyle is key to preventing future mental health concerns."

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse input data from the request
        data = request.json
        new_data = np.array([
            [
                data['gender'],
                data['age'],
                data['work_pressure'],
                data['job_satisfaction'],
                data['sleep_duration'],
                data['dietary_habits'],
                data['suicidal_thoughts'],
                data['work_hours'],
                data['financial_stress'],
                data['family_history'],
            ]
        ])
        # Make prediction
        prediction = lr_classifier.predict(new_data)
        probability = lr_classifier.predict_proba(new_data)[0][1] # Make sure this line is correct
        description = get_probability_description(probability)
        risk_level = get_risk_level(probability)
        return jsonify({"description": description, "probability": probability, "risk_level": risk_level}) # probability MUST be included here

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)