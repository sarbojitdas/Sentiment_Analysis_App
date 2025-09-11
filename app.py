from flask import Flask, render_template, request
import pickle

# Load model and vectorizer
model = pickle.load(open('sentiment_model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    text_vector = vectorizer.transform([text])
    prediction = model.predict(text_vector)[0]
    sentiment = "Positive 😊" if prediction == 1 else "Negative 😞"
    return render_template('index.html', prediction=sentiment, input_text=text)

if __name__ == '__main__':
    app.run(debug=True)
