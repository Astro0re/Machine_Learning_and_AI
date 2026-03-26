# linear Regression

# Packages
import pandas as pd
import numpy as np 
import matplotlib as plts
import tensorflow as tf
from sklearn.linear_model import LinearRegression

df =pd.read_csv('iris.csv')

x = df.drop(columns=['id','species'])
y= df['species']

model = LinearRegression()

model.fit(x,y)

model.predict([[5.1,3.5,1.4,0.2]])

plts.scatter(df['sepal_length'],df['sepal_width'],c=df['species'])
