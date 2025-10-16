# Student LLM 
#Large language model based on digital documents of my course work during college
#Start with a single course first then scale to semeter...textbooks etc"
# Packages 
import os
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import requests
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler
import keras 
from kera.preprocessing import Tokenizer 

# Web Scraping Function 
def scrape():
    global soup
    topic= input("What topic do you want to learn? ")
    scr= requests.get(f"https://en.wikipedia.org/wiki/{topic}")
    soup = BeautifulSoup(scr.text, "html.parser")
    if scr is True:
        print("Details collected succesfully")
    else:
        print("Details not collected")
    return soup



# Web Scrape the topics 
corpus = pd.read_html("C:/Users/USER/Documents/VSC/Git_/AI/Course_Work_LLM/index.html")
    
text =scrape(corpus)


# Tokenizer 
tokenized = tf.keras.preprocessing.Tokenizer(text)

# Encoder (Character to Integer)
for i in range(len(corpus)):
    if [i] == "A":
        print ("True")
    
encode = LabelEncoder()

vectors = encode.fit_transform(tokenized)

model = tf.keras.Transformer

LLM = model.train(vectors)
# Decoder (Integer to Character)