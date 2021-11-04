from flask import Flask,render_template,request
import joblib

# initializing the app
app = Flask(__name__)

# loading model
note_model = joblib.load('bank_note_99.pkl')
sc = joblib.load('note_standard.pkl')


@app.route('/')
def hello():
    return render_template('form.html')


@app.route('/predict',methods=["GET","POST"])
def predict():
    
    VARIANCE = request.form.get('VARIANCE')
    SKEWNESS = request.form.get('SKEWNESS')
    CURTOSIS = request.form.get('CURTOSIS')
    ENTROPY = request.form.get('ENTROPY')

    data = [[VARIANCE,SKEWNESS,CURTOSIS,ENTROPY]]
    my_pred = note_model.predict(sc.transform(data))

    if my_pred == [1]:
        out = 'Real Note'
    else:
        out = 'Fake or Mutilated Note'

    return render_template('predict.html', prediction = f'{out}')



if __name__ == '__main__':
    app.run(debug=True)