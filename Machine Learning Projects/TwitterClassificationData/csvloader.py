import numpy as np
import pandas as pd
import re
from sklearn.model_selection import train_test_split
#Converting text to num
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
def load_data(csv,enco):
    df=pd.read_csv(csv,encoding=enco,header=0,converters={"sidebar_color": lambda x: int(x, 16)})
    daf=df[df['gender:confidence']==1]
    daf=daf[daf['gender'].isin(['male','female'])]
    return daf
    
def loadcsv(filename):
    dataset = np.genfromtxt(filename, delimiter=',',names=True)
    return dataset

def normalize_data(daf):
    def normalstring(strg):
        s=str(strg)
        s=s.lower()
        s=re.sub('\s\W',' ',s)
        s=re.sub('\W\s',' ', s)
        s=re.sub('\s+',' ',s)
        return s
    daf['text_norm']=[normalstring(strin) for strin in daf['text']]
    daf['description_norm']=[normalstring(strin) for strin in daf['description']]
    
    return daf

def seperatey(daf):
    '''
    vector=CountVectorizer()
    x=vector.fit_transform(daf['text_norm'])
    '''
    x= daf['tweet_count']
    encoder=LabelEncoder()
    y=encoder.fit_transform(daf['gender'])
    xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3)
    return xtrain,ytrain,xtest,ytest
    
    