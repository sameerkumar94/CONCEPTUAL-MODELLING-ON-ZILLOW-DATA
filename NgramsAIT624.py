import pandas as pd
import re
import numpy as np
import nltk
from nltk import word_tokenize
from nltk.util import ngrams
from collections import Counter
import string
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import collections
import sys
import time
from nltk.collocations import BigramCollocationFinder
from nltk.collocations import TrigramCollocationFinder
from nltk.collocations import *
import spacy

#%%%%%%%%%%%%%%%%%%%%%%%
data=pd.read_csv("C:/Users/csame/Desktop/AIT 624 Project/624project.csv",encoding='ISO-8859-1')
#%%%%%%%%%%%%%%%%%%%%
ps=nltk.WordNetLemmatizer()

stopwords=set(stopwords.words('english'))
stopwords=list(stopwords)
Location=pd.read_csv("C:/Users/csame/Desktop/AIT 624 Project/Location.csv")
Loc=[]
for i in range(len(Location)):
   Loc.append(str(Location['City'].iloc[i]).lower())
   stopwords.append((Location['City'].iloc[i]).lower())
   stopwords.append((Location['City'].iloc[i]).lower()+'va')
#%%%%%%%%%%
   
stopword_more=['u','va','dr','zillow','state','ter','id','k','x','e','websitesee','z','data','zillows','code','le','dont','lo','mi','/'
               ,'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','manassasva','manassas'
              ,'aldie', 'manassas', 'middleburg', 'centreville', 'clifton', 'bristow', 'ashburn', 'chantilly', 'gainesville', 'sterling', 'herndon', 'leesburg',
              'aldieva', 'manassasva', 'middleburgva', 'centrevilleva', 'cliftonva', 'bristowva', 'ashburnva', 'chantillyva', 'gainesvilleva', 'sterlingva', 'herndonva', 'leesburgva']
for i in range(len(stopword_more)):
    stopwords.append(stopword_more[i])
    
stopwords=set(stopwords)    


data.columns = ['','location','condodetail']

data_needed=pd.DataFrame({'location':data['location'],
                   'condodetail':data['condodetail']})
#print(data_needed)
def rem_stopwords(text):
    text=re.sub('{.*?}', '', text)
    tokens= re.split('\W+',text)
    text=" ".join([word for word in tokens if word not in stopwords])
    text= "".join([word for word in text if word not in string.punctuation])
    text=" ".join([word for word in tokens if word not in stopwords])
    return text

data_needed['Cdetail']=data_needed['condodetail'].apply(lambda x: rem_stopwords(x.lower()))
print(data_needed)
#%%%%
for i in range(len(data_needed)):
    data_needed['location'].iloc[i]=re.sub('-va','',str(data_needed['location'].iloc[i]))
    data_needed['location'].iloc[i]=re.sub('bull-run-','',str(data_needed['location'].iloc[i]))
    #data_needed['location'].iloc[i]=re.sub(r'[^\]|]', r'', data_needed['location'].iloc[i])
    #data_needed['Loc'].iloc[i]=re.sub("]'","",str(data_needed['Loc'].iloc[i]))
    #data_needed['location'].iloc[i]=data_needed['location'].iloc[i].replace("'","")
#print(data_needed)
#%%%%%%%%%%%%%%%%%%%%%%
Location=pd.read_csv("C:/Users/csame/HealthDisasterNLP/CondoProject624/Location.csv")
Loc=[]
for i in range(len(Location)):
   Loc.append(str(Location['City'][i]).lower())

def cities(text):
    
    
    tokens= re.split('\W+',str(text))
    text=" ".join([word for word in tokens if word in Loc])
    tokens1= re.split(' ',text)
    
    return tokens1[0] 

data_needed['Location']=data_needed['location'].apply(lambda x: cities(x.lower()))

print(data_needed)
#%%%%%%%%%%%%%%%%%
Data_clean=data_needed[['Location',
 'Cdetail']].copy()
print(Data_clean)
Data_clean.to_csv("C:/Users/csame/HealthDisasterNLP/CondoProject624/Condofinal.csv")
#%%%%%%%%%%%
city = []
for sent in Data_clean['Location']:
    city.append(sent)

output = []
for x in city:
    if x not in output:
        output.append(x)
print (output)

Combined_data=pd.DataFrame({'Address':[],'Text':[]})
k=''
for i in range(len(output)):
    k=''
    for j in range(len(Data_clean)):
        if str(output[i]) == str(Data_clean['Location'][j]):
            k=k+' '+Data_clean['Cdetail'].iloc[j]              
    Combined_data=Combined_data.append({'Address':str(output[i]),'Text':str(k)},ignore_index=True)
#print(Data_clean)
#%%%%%%%%%%%
def bi(text):
    bigram_measures = nltk.collocations.BigramAssocMeasures()
    finder=BigramCollocationFinder.from_words(word_tokenize(text))
    finder.apply_freq_filter(20)
    finder.nbest(bigram_measures.pmi, 50) 
    print(finder.ngram_fd.items())
    print(len(finder.ngram_fd.items()))
    return finder.ngram_fd.items()
Combined_data['Bi']=Combined_data['Text'].apply(lambda x: bi(x.lower()))
#print(finder.ngram_fd.items())
#print(Combined_data['Bi'].iloc[0])
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def tri(text):
    trigram_measures = nltk.collocations.TrigramAssocMeasures()
    finder=TrigramCollocationFinder.from_words(word_tokenize(text))
    finder.apply_freq_filter(30)
    finder.nbest(trigram_measures.pmi, 200) 
    print(finder.ngram_fd.items())
    print(len(finder.ngram_fd.items()))
    return finder.ngram_fd.items()

Combined_data['Tri']=Combined_data['Text'].apply(lambda x: tri(x.lower()))
#%%%%%

nlp=spacy.load('en')
Combined_data['Tri_similarity']=''
for i in range(len(Combined_data['Tri'])):
    text=[]
    Similarities=[]
    for j in range(len(list(Combined_data['Tri'].iloc[i]))):
        x=list(Combined_data['Tri'].iloc[i])
        y=list(x[j][0])
        words=''
        for l in range(len(y)):
            words=words+y[l]+" "
        text.append(words.strip())
    similarity=[]
    for k in range(len(text)-1):
        for m in range(k+1,len(text)):
            doc=nlp(text[k])
            doc1=nlp(text[m])
            test=doc.similarity(doc1)
            similarity.append(test)
            similarity.append(text[k])
            similarity.append(text[m])
        Similarities.append(similarity)
    Combined_data['Tri_similarity'].iloc[i]=Similarities    
#%%%%%%%%%%%%%%%
Combined_data['Bi_similarity']=''
for i in range(len(Combined_data['Bi'])):
    text=[]
    Similarities=[]
    for j in range(len(list(Combined_data['Bi'].iloc[i]))):
        x=list(Combined_data['Bi'].iloc[i])
        y=list(x[j][0])
        words=''
        for l in range(len(y)):
            words=words+y[l]+" "
        text.append(words.strip())
    similarity=[]
    for k in range(len(text)-1):
        for m in range(k+1,len(text)):
            doc=nlp(text[k])
            doc1=nlp(text[m])
            print(text[k],text[m])
            test=doc.similarity(doc1)
            similarity.append(test)
            similarity.append(text[k])
            similarity.append(text[m])
        Similarities.append(similarity)
    Combined_data['Bi_similarity'].iloc[i]=Similarities
#%%%%%%%%%%%%%%%%%%%%%%5
Combined_data.to_csv('C:/Users/csame/Desktop/AIT 624 Project/Condo_final.csv')





















