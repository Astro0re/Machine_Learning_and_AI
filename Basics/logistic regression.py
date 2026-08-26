# logistic regression

# Packages 
import os 
import pandas as pd
import numpy as np 
import matplotlib.pyplot as py
from sklearn.linear_model import LogisticRegression

df =pd.read_csv('iris.csv')

x = df.drop(columns=['id','species'])
y= df['species']

model = LogisticRegression(class_weight='balanced', max_iter= 1000, random_state=42)

model.fit(x,y)

model.predict([[5.1,3.5,1.4,0.2]])

py.scatter(df['sepal_length'],df['sepal_width'],c=df['species'])