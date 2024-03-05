import re
import webvtt
from gensim.summarization.summarizer import summarize as gensim_based
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import nltk
from tkinter import *
from tkinter import filedialog
import tkinter.font as tkFont
import os
import uvicorn
import fastapi
import youtube_dl
import pandas as pd
import numpy as np
import spacy
nlp = spacy.load("en_core_web_sm")

####################################################################################
# Function Block


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")



@app.get('/')
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

def get_caption(url):
    global video_title
    # Using Youtube-dl inside python
    ydl_opts = {
        'skip_download': True,        # Skipping the download of actual file
        'writesubtitles': True,       # Uploaded Subtitles
        "writeautomaticsub": True,    # Auto generated Subtitles
        "subtitleslangs": ['en'],     # Language Needed "en"-->English
        'outtmpl': 'test.%(ext)s',    # Saving downloaded file as 'test.en.vtt'
        'nooverwrites': False,        # Overwrite if the file exists
        'quiet': True                # Printing progress
    }

    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        try:
            ydl.download([url])
            info_dict = ydl.extract_info(url, download=False)
            video_title = info_dict.get('title', None)
        except:
            print("Try with a YouTube URL")
    corpus = []
    for caption in webvtt.read('test.en.vtt'):
        corpus.append(caption.text)
    corpus = "".join(corpus)
    corpus = corpus.replace('\n', ' ')

    return corpus


def summarizer(text, option, fraction):
    # "Tf-IDF-Based", "Frequency-Based", "Gensim-Based"
    
    frac=fraction
    if option == "TfIdf-Based":
        return tfidf_based(text, frac)
    if option == "Frequency-Based":
        return freq_based(text, frac)
    if option == "Gensim-Based":
        doc=nlp(text)
        text="\n".join([sent.text for sent in doc.sents])
        return gensim_based(text=text, ratio=frac)

def tfidf_based(msg,fraction=0.3):
    # Creating Pipeline
    doc=nlp(msg)
    
    #Sent_tokenize
    sents =[sent.text for sent in doc.sents]
    
    #Number of Sentence User wants
    num_sent=int(np.ceil(len(sents)*fraction))
    
    #Creating tf-idf removing the stop words matching token pattern of only text
    tfidf=TfidfVectorizer(stop_words='english',token_pattern='(?ui)\\b\\w*[a-z]+\\w*\\b')
    X=tfidf.fit_transform(sents)
    
    #Creating a df with data and tf-idf value
    df=pd.DataFrame(data=X.todense(),columns=tfidf.get_feature_names())
    indexlist=list(df.sum(axis=1).sort_values(ascending=False).index)
#     indexlist=list((df.sum(axis=1)/df[df>0].count(axis=1)).sort_values(ascending=False).index)
    
    # Subsetting only user needed sentence
    needed = indexlist[:num_sent]
    
    #Sorting the document in order
    needed.sort()
    
    #Appending summary to a list--> convert to string --> return to user
    summary=[]
    for i in needed:
        summary.append(sents[i])
    summary="".join(summary)
    summary = summary.replace("\n",'')
    return summary


def freq_based(text, fraction):
    # Convert to pipeline
    doc = nlp(text)
    # Break to sentences
    sentence = [sent for sent in doc.sents]
    # Number of sentence user wants
    numsentence = int(np.ceil(fraction*len(sentence)))

    # Tokenizing and filtering key words
    words = [word.text.lower()
             for word in doc.doc if word.is_alpha and word.is_stop == False]
    # Converting to df for calculating weighted frequency
    df = pd.DataFrame.from_dict(
        data=dict(Counter(words)), orient="index", columns=["freq"])
    df["wfreq"] = np.round(df.freq/df.freq.max(), 3)
    df = df.drop('freq', axis=1)

    # Convert weighted frequency back to dict
    wfreq_words = df.wfreq.to_dict()

    # Weight each sentence based on their wfreq
    sent_weight = []
    for sent in sentence:
        temp = 0
        for word in sent:
            if word.text.lower() in wfreq_words:
                temp += wfreq_words[word.text.lower()]
        sent_weight.append(temp)
    wdf = pd.DataFrame(data=np.round(sent_weight, 3), columns=['weight'])
    wdf = wdf.sort_values(by='weight', ascending=False)
    indexlist = list(wdf.iloc[:numsentence, :].index)

    # Summary
    sumlist = []
    for s in indexlist[:5]:
        sumlist.append(sentence[s])
    summary = ''.join(token.string.strip() for token in sumlist)
    return summary   
if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)