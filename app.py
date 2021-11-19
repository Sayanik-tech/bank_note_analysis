from flask import Flask,render_template,request
import joblib

# initializing the app
app = Flask(__name__)

# loading model
knn_classifier = joblib.load('bank_note_99.pkl')
sc = joblib.load('note_standard.pkl')


@app.route('/')
def hello():
    return render_template('form.html')


@app.route('/predict',methods=["GET","POST"])
def predict():
    
    variance = request.form.get('variance')
    skewness = request.form.get('skewness')
    curtosis = request.form.get('curtosis')
    entropy = request.form.get('entropy')

    
    data = [[variance,skewness,curtosis,entropy]]
    knn_pred = knn_classifier.predict(sc.transform(data))

    

    if knn_pred[0] == 1:
        out = 'Real Note'
    else:
        out = 'Fake or Mutilated Note'
    
    
    return render_template('predict.html', prediction = f'{out}')
    



if __name__ == '__main__':
    app.run(debug=True)