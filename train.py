import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold,GridSearchCV
from sklearn.model_selection import cross_val_score

# importing data set
dataset = pd.read_csv('https://github.com/Sayanik-tech/bank_note_analysis/raw/main/BankNote_Authentication%2B(1)%20(1).csv')
print(dataset.head())

X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

# Splitting the data

from sklearn.model_selection import train_test_split
x_train,x_test,y_yrain,y_test = train_test_split(X,y,test_size = 0.2,random_state = 1)

# Scaling the data

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# KNN Classifier

parameters_dict = {'n_neighbors':range(2,15), 'leaf_size':range(2,15), 'p':[1,2]}
grid = GridSearchCV( estimator = KNeighborsClassifier(),param_grid = parameters_dict,n_jobs=-1,cv=10,verbose=1,)

grid.fit(x_train,y_yrain)

print(grid.best_params_)
print(grid.best_score_)

#fitting model
knn_classifier = KNeighborsClassifier(n_neighbors=8,p=1,leaf_size=2)
knn_classifier.fit(x_train,y_yrain)

# accuracy score
knn_accuracies = cross_val_score(estimator = knn_classifier,X = x_train,y = y_yrain, cv = 10)
print('Accuracies:\n', knn_accuracies)
print("Accuracy: {:.2f} %" .format(knn_accuracies.mean()*100))
print("Standard Deviation: {:.2f} %" .format(knn_accuracies.std()*100))

#saving model

import joblib
joblib.dump(knn_classifier,'bank_note_99.pkl')
joblib.dump(sc,'note_standard.pkl')





