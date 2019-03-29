import re
import pandas as pd
import nltk
from nltk import word_tokenize
from nltk.util import ngrams
from collections import Counter
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import CountVectorizer
import collections
import sys
import time
from nltk.collocations import BigramCollocationFinder
from nltk.collocations import TrigramCollocationFinder
from nltk.collocations import *
import spacy

#%%%%%%%%%%%%%%%%%%%%%%%
data=pd.read_csv("C:/Users/csame/Desktop/AIT 624 Project/Condo_Virginia_Cleaned TXT.csv",header=None,encoding='ISO-8859-1')
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
    #%%%%%%%%%%%%%%%%%%%%%


stopword_more=['u','va','dr','zillow','state','ter','id','k','x','e','websitesee','z','data','zillows','code','le','dont','lo','mi','/'
               ,'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','manassasva','manassas'
              ,'aldie', 'manassas', 'middleburg', 'centreville', 'clifton', 'bristow', 'ashburn', 'chantilly', 'gainesville', 'sterling', 'herndon', 'leesburg',
              'aldieva', 'manassasva', 'middleburgva', 'centrevilleva', 'cliftonva', 'bristowva', 'ashburnva', 'chantillyva', 'gainesvilleva', 'sterlingva', 'herndonva', 'leesburgva']
for i in range(len(stopword_more)):
    stopwords.append(stopword_more[i])
    

stopwords=set(stopwords)


data.columns = ['AD','Post','Owner_desc','ADD_com','Reviews']

data_needed=pd.DataFrame({'AD':data['AD'],
                   'full_DATA':data['Reviews']})
def rem_stopwords(text):
    text=re.sub('{.*?}', '', text)
    tokens= re.split('\W+',text)
    text=" ".join([word for word in tokens if word not in stopwords])
    text= "".join([word for word in text if word not in string.punctuation])
    text=" ".join([word for word in tokens if word not in stopwords])
    return text

data_needed['cleantext']=data_needed['full_DATA'].apply(lambda x: rem_stopwords(x.lower()))
#%%%%%%%%%%%%%%   
def cities(text):
    
    
    tokens= re.split('\W+',str(text))
    text=" ".join([word for word in tokens if word in Loc])
    tokens1= re.split(' ',text)
    
    return tokens1[0] 

data_needed['cleantext_AD']=data_needed['AD'].apply(lambda x: cities(x.lower()))

#%%%%%%%%%%%%%%%%%
Data_clean=data_needed[['cleantext',
 'cleantext_AD']].copy()
#%%%%%%%%%%%
Combined_data=pd.DataFrame({'AD':[],'full_DATA':[]})
print(Combined_data)
#%%%%%%%%%%%%%
city = []
for sent in Data_clean['cleantext_AD']:
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
        if str(output[i]) == str(Data_clean['cleantext_AD'][j]):
            k=k+' '+Data_clean['cleantext'].iloc[j]              
    Combined_data=Combined_data.append({'Address':str(output[i]),'Text':str(k)},ignore_index=True)
#%%%%%%%%%%%%%%%%%
def bi(text):
    bigram_measures = nltk.collocations.BigramAssocMeasures()
    finder=BigramCollocationFinder.from_words(word_tokenize(text))
    finder.apply_freq_filter(30)
    finder.nbest(bigram_measures.pmi, 20) 
    return finder.ngram_fd.items()

Combined_data['Bi']=Combined_data['Text'].apply(lambda x: bi(x.lower()))
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def tri(text):
    trigram_measures = nltk.collocations.TrigramAssocMeasures()
    finder=TrigramCollocationFinder.from_words(word_tokenize(text))
    finder.apply_freq_filter(30)
    finder.nbest(trigram_measures.pmi, 20) 
    return finder.ngram_fd.items()

Combined_data['Tri']=Combined_data['Text'].apply(lambda x: tri(x.lower()))    
    
#%%%%%%%%%%%%%%%%%%
import spacy
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
