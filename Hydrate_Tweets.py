# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 13:03:55 2022

@author: jonpr
"""

#### Tweet Hydrator

import os
import pandas as pd
from twarc import Twarc2, Twarc

# paths
os.chdir(r"C:\Users\jonpr\Documents\Data projects\Python\TwitterConspiracy")
path = os.getcwd()
dataPath = os.path.join(path, "WICO")

# create tweet id file
ids = pd.DataFrame()
for root,dirs,files in os.walk(dataPath):
    for file in files:
       if file.endswith(".csv"):
           df = pd.read_csv(os.path.join(root, file))
           ids = pd.concat([ids, df])
ids = ids.reset_index(drop=True)['id']

# load twitter API credentials
API_info = pd.read_csv('API_Credentials.csv')
t = Twarc(API_info['consumer_key'][0], API_info['consumer_secret'][0], API_info['access_token'][0], API_info['access_token_secret'][0])
t2 = Twarc2(bearer_token=API_info['bearer_token'][0])
 
# Hydrate and save tweets
tweetText = []        
for tweet in t.hydrate(ids):
    tweetText.append(tweet['full_text'])

tdf = pd.DataFrame(tweetText, columns=["tweetText"])
tdf.to_csv('tweetText.csv', index=False)