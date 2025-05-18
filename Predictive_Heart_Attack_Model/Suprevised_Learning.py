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
from Predictive_Heart_Attack_Model.Suprevised_Learning import organize_col

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


 # Select columns by dtype using pandas select_dtypes
def organize_col(df):
    factor_int = df.select_dtypes(include=['int64', 'int32', 'int16', 'int8','float64'])
    factor_bool = df.select_dtypes(include=['bool'])
    factor_char = df.select_dtypes(include=['object', 'string'])

    return factor_int, factor_bool, factor_char

organize_col(factor)
factor_seperated=organize_col(factor)

Int_factor = factor_seperated['factor_int']
bool_factor = factor_seperated['factor_bool']

label_encoders = {}
    for column in :
        label_encoders[column] = LabelEncoder()
        heart_data[column] = label_encoders[column].fit_transform(heart_data[column])
        print(f"{column} unique values after encoding:", heart_data[column].unique())

# X= Heart_Data["Obesity", "Heart-Attack-History","Cholesterol","Diet",]

target = Heart_Data["Heart_Attack"]
print(target)


kn=KNeighborsRegressor()
kn.fit(factor,target)
pre_1=kn.predict(factor)
print(pre_1)

lr=LinearRegression()
lr.fit(factor,target)
pre_2 = lr.predict(factor)
print(pre_2)