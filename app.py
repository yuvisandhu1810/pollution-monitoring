from flask import Flask, render_template, request
import joblib
import os

app = Flask(__name__)

# Load the model
model = joblib.load('pollution_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [float(x) for x in request.form.values()]
        prediction = model.predict([features])
        return render_template('index.html', prediction_text=f'Predicted Air Quality: {prediction[0]}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))   # <- Important change
    app.run(host='0.0.0.0', port=port)          # <- Important change
