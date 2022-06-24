import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from naive_bayes_classifier import NaiveBayes


df = pd.read_csv('Naive_Bayes_Classifier/spam_dataset.csv')
X = df.iloc[:,:-1]
y=df.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X,y,train_size = 0.8, random_state=42)

nb = NaiveBayes()
nb.fit(X_train, y_train)
y_pred = nb.predict(X_test)
print(y_pred)