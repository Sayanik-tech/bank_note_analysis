import joblib

knn_classifier = joblib.load('bank_note_99.pkl')
sc = joblib.load('note_standard.pkl')

data = [[-4.1409,3.4619,-0.47841,-3.8879]]

knn_pred = knn_classifier.predict(sc.transform(data))

if knn_pred[0] == 1:
    print('Real Note')
else:
    print('Fake or Mutilated Note')