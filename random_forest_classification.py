## importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing the Dataset

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
 
# Assign colum names to the dataset
column_names = ['Sepal-length', 'Sepal-width', 'Petal-length', 'Petal-width', 'Class']
 
# Reading dataset 
data = pd.read_csv(url, names=column_names)
# print(data)

X=data.iloc[:,:4].values  # Independent Variable
Y=data.iloc[:,4].values   # Dependent Variable

## splitting dataset into training and testing set

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25,random_state=0)
# print(X_train)
# print(X_test)
# print(Y_train)
# print(Y_test)

## feature scaling

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)
# print(X_train)
# print(X_test)

## training the Random Forest Classification model on the training set

from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=5,criterion='entropy',random_state=0)
classifier.fit(X_train,Y_train)

## predicting the test set result

Y_pred = classifier.predict(X_test)
# print(np.concatenate((Y_pred.reshape(len(Y_pred),1),Y_test.reshape(len(Y_test),1)),axis=1))


## making the confusion matrix(no. of incorrect and correct predictions)

from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(Y_test,Y_pred)
print(cm)
print(accuracy_score(Y_test,Y_pred))