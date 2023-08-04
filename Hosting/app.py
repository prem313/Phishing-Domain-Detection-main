from flask import Flask, request, Response, render_template
import pickle
import os

app = Flask(__name__)

# Specify the absolute file path to the pickle file
model_file_path = r"C:\Users\DELL\WORKSPACE\PROJECTS\Phishing-Domain-Detection-main\Hosting\phishing.pkl"

# Load the pickled model from the file
phish_model_ls = pickle.load(open(model_file_path, 'rb'))

urlError = {
    "Please enter url field"
}

# Rest of your code remains the same...



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',  methods=['POST'])
def predict():

    X_predict = []

    url = request.form.get("EnterYourSite")
    print(url, "0000000000000000000000")
    if url:
        X_predict.append(str(url))
        y_Predict = ''.join(phish_model_ls.predict(X_predict))
        print(y_Predict)
        if y_Predict == 'bad':
            result = "This is a Phishing Site"
        else:
            result = "This is not a Phishing Site"

        return render_template('index.html', prediction_text = result)

    elif not url:
        return Response(
            response=urlError,
            status=400
        )

if __name__ == '__main__':

    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
