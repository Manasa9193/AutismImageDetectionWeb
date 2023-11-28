from flask import Flask, render_template, request
from test import load_and_predict

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def make_prediction():
    if request.method == 'POST':
        image_file = request.files['image']
        image_path = f"uploads/{image_file.filename}"
        image_file.save(image_path)

        prediction = load_and_predict(image_path)

        result = "Autistic" if prediction == 0 else "Not Autistic"
        return render_template('index.html', prediction=result, image_file=image_file.filename)

if __name__ == '__main__':
    app.run(debug=True)
