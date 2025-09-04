# Student LLM 
#Large language model based on digital documents of my course work during college
#Start with a single course first then scale to semeter...textbooks etc"
# Packages 
import pandas as pd
import numpy as np 
import matplotlib as plts
import beautifulsoup4 as bs4
import requests
import tensorflow as tf

# Web Scraping Function 
def scrape():
    topic= input("What topic do you want to learn? ")
    scr= pd.read_html(f"https://en.wikipedia.org/wiki/{topic}")
    if scr is True:
        print("Details collected succesfully")
    return scr
  
   
# Web Scrape the topics 
corpus = pd.read_html("/c/Users/Data Analyst/Documents/GitHub/Fesobi-Oreofe/index.html")
    
scrape()


# Tokenizer 
  
# Encoder (Character to Integer)
for i in range(len(corpus)):
    if [i] == "A":
        print ("True")
    
# Decoder (Integer to Character)