# Basic Packages
import urllib
import statistics
from statistics import linear_regression

import numpy as nu
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# To use Heart Attack in Russia Data to create a predictive model

# X being the factors to base the prediction on
# X Factors = Heart Attack History, Cholesterol, Diet, Excercise Habits, Obesity and others

# Y being the factor to predict (prediction)



# Data Exploration
Heart_Data =pd.read_csv ( r"C:\Users\USER\Documents\VSC\Git_\AI\Predictive_Heart_Attack_Model\heart_attack_russia.csv" )
print(Heart_Data.info())
print(Heart_Data.head())
print(Heart_Data.shape)
# print(Heart_Data.index)

Heart_Data = Heart_Data.fillna('Null')

factor = Heart_Data.loc[:, Heart_Data.columns != 'Heart_Attack']

print(factor)

type(factor.column).value_counts()

def organize_col(factor):
    factor_int=pd.DataFrame()
    factor_char=pd.DataFrame()
    factor_bool=pd.DataFrame()

    for column in factor.columns:
        if factor.column.astype(int) == True:
            factor_int.append

        if factor.column.astype(chr) == True :
            factor_char.append

        if factor.column.astype(bool) == True :
            factor_bool.append


label_encode= LabelEncoder()
factor[column]= label_encode[column]
fit_transform(factor[column])


# X= Heart_Data["Obesity", "Heart-Attack-History","Cholesterol","Diet",]

target = Heart_Data["Heart_Attack"]
print(Y)


kn=KNeighborsRegressor()
kn.fit(X,Y)
pre_1=kn.predict(X)
print(pre_1)

lr=LinearRegression()
lr.fit(X,Y)
pre_2 = lr.predict(X)
print(pre_2)