# Student LLM 
#Large language model based on digital documents of my course work during college
#Start with a single course first then scale to semeter...textbooks etc"
# Packages 
import pandas as pd
import numpy as np 
import matplotlib as plts
from bs4 import BeautifulSoup
import requests
import tensorflow as tf

# Web Scraping Function 
def scrape():
    topic= input("What topic do you want to learn? ")
    scr= requests.get(f"https://en.wikipedia.org/wiki/{topic}")
    soup = BeautifulSoup(scr.text, "html.parser")
    if scr is True:
        print("Details collected succesfully")
    else:
        print("Details not collected")
    return soup
  
   
# Web Scrape the topics 
corpus = pd.read_html("/c/Users/USER/Documents/VSC/Git_/AI/Course_Work_LLM/index.html")
    
scrape()


# Tokenizer 
tf.keras.preprocessing.text.Tokenizer()

# Encoder (Character to Integer)
for i in range(len(corpus)):
    if [i] == "A":
        print ("True")
    
# Decoder (Integer to Character)