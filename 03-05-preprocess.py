# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 10:35:13 2024

@author: huqio
"""

import pandas as pd
# read the clean data
df=pd.read_csv("")


# find non empty comments 

df1= df[df["comment"].str.len() == 1]

# make sure no duplicates
df1_uniq=df1.comment.unique()

# Merge df and df1 with indicator argument
merged = pd.merge(df, df1, how='outer', indicator=True)

# Filter out rows only present in df
rows_only_in_df = merged[merged['_merge'] == 'left_only'].drop(columns='_merge')
df1=rows_only_in_df

# no comments for provider
df1= df[df[""].str.len() > 0]

# put all the reviews with comments into a new file
df1.to_excel(".xlsx")




######################################################################################
# split comments into sentence and assign new index and remove all stopwords and sentiments words
import pandas as pd
from nltk import word_tokenize
import re  

def clean_text(text):
    # Clean text by replacing invalid characters with an empty string
    cleaned_text = ''.join([c if ord(c) < 128 else '' for c in text])
    return cleaned_text

    
with open("stopwords.txt", 'r', encoding='utf-8') as stopword_file:
    stop_words = stopword_file.read().splitlines()
stop_words = set(stop_words)

# just for those pods which need to combine the comment for the provider and the experience
df = pd.read_excel("")


## easy method to split sentence
# from nltk.tokenize import sent_tokenize
#nltk.tokenize.sent_tokenize(text, language='english')[source]


# lower words
df.sentence=df.sentence.astype(str)
df["sentence_lower"]=df.sentence.map(lambda x: x.lower()).tolist()


# remove punctuations
import string
import numpy as np
df["sentence_lower_remPun"]=df.sentence_lower.map(lambda x:x.translate(str.maketrans('', '', string.punctuation))).tolist()
df["sentence_lower_remPun"] = df["sentence_lower_remPun"].replace(r'^\s+$', np.nan, regex=True)


############# Apply the function to the column
# replace numbers as empty string
df['sentence_lower_remPunNum'] = df['sentence_lower_remPun'].str.replace('\d+', '',regex=True)
# Replace empty spaces with empty strings in the specified column
df['sentence_lower_remPunNum'] = df['sentence_lower_remPunNum'].replace(r'^\s*$', '', regex=True)
df['sentence_lower_remPunNum']=df['sentence_lower_remPunNum'].replace(np.nan, '')

language=[]

for text in df["sentence_lower_remPunNum"]:
    if len(text) == 0:
        temp=text
    elif detect(text)=="en":
        # print(text)
        temp=text
    else:
        temp=GoogleTranslator(source=detect(text), target='en').translate(text)
    language += [temp]  
    
df["sentence_lower_removePunNum_en"]=language




from nltk.tokenize import word_tokenize
# nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
stop_words = stopwords.words('english')
# print(stop_words)

def remove_stopwords(texts):
    text_tokens=word_tokenize(texts)
    tokens_without_sw = [word for word in text_tokens if word not in stop_words]
    return " ".join( tokens_without_sw)


with open('stopwords.txt', encoding="utf8") as f:
    lines = f.readlines()
    adv_stop_words = lines
        
def remove_advance_stopwords(texts):
    text_tokens=word_tokenize(texts)
    tokens_without_sw = [word for word in text_tokens if word not in adv_stop_words]
    return " ".join( tokens_without_sw)

df.sentence_lower_removePunNum_en=df.sentence_lower_removePunNum_en.astype(str)

data = df.sentence_lower_removePunNum_en.values.tolist()

result_eng=[remove_stopwords(i) for i in data]

result_advance=[remove_advance_stopwords(i) for i in result_eng]

df=df.assign(sentence_lower_removePunNum_en_eng=result_eng)

df=df.assign(sentence_lower_removePunNum_en_eng_adv=result_advance)
  
df["sentence_lower_removePunNum_en_eng_adv_blank"]=df["sentence_lower_removePunNum_en_eng_adv"].str.replace('blank','')

###remove sentiment words
df_st=pd.read_excel("sentiment_words.xlsx")
senti_words=df_st["words"].unique().tolist()


def remove_sentiment_words(texts):
    text_tokens=word_tokenize(texts)
    tokens_without_sw = [word for word in text_tokens if word not in senti_words]
    return " ".join( tokens_without_sw)

data_senti=df.sentence_lower_removePunNum_en_eng_adv_blank.values.tolist()

result_sentiment = [remove_sentiment_words(i) for i in data_senti]

df=df.assign(sentence_lower_removePunNum_en_eng_adv_blank_senti=result_sentiment)

# df.to_excel("temp_4000.xlsx")

# lemmatize the words by finding the type of the word first
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
class Splitter(object):
    """
    split the document into sentences and tokenize each sentence
    """
    def __init__(self):
        self.splitter = nltk.data.load('tokenizers/punkt/english.pickle')
        self.tokenizer = nltk.tokenize.TreebankWordTokenizer()

    def split(self,text):
        """
        out : ['What', 'can', 'I', 'say', 'about', 'this', 'place', '.']
        """
        # split into single sentence
        sentences = self.splitter.tokenize(text)
        # tokenization in each sentences
        tokens = [self.tokenizer.tokenize(sent) for sent in sentences]
        return tokens


class LemmatizationWithPOSTagger(object):
    def __init__(self):
        pass
    def get_wordnet_pos(self,treebank_tag):
        """
        return WORDNET POS compliance to WORDENT lemmatization (a,n,r,v) 
        """
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            # As default pos in lemmatization is Noun
            return wordnet.NOUN

    def pos_tag(self,tokens):
        # find the pos tagginf for each tokens [('What', 'WP'), ('can', 'MD'), ('I', 'PRP') ....
        pos_tokens = [nltk.pos_tag(token) for token in tokens]
        # lemmatization using pos tagg   
        # convert into feature set of [('What', 'What', ['WP']), ('can', 'can', ['MD']), ... ie [original WORD, Lemmatized word, POS tag]
        pos_tokens = [ [(word, lemmatizer.lemmatize(word,self.get_wordnet_pos(pos_tag)), [pos_tag]) for (word,pos_tag) in pos] for pos in pos_tokens]
        return pos_tokens


lemmatizer = WordNetLemmatizer()
splitter = Splitter()
lemmatization_using_pos_tagger = LemmatizationWithPOSTagger()


data_lem=df.sentence_lower_removePunNum_en_eng_adv_blank_senti.values.tolist()
# test_text=("Below is the implementation of lemmatization words", "I learned today")
# new_list=[]
# for i in data_lem:
def lem_text(text):    
    tokens = splitter.split(text)
    lemma_pos_token = lemmatization_using_pos_tagger.pos_tag(tokens)
    # print(lemma_pos_token)
    a=lemma_pos_token[0]
    b=[x[1] for x in a]
    return " ".join(b)


temp_list=[]
for i in data_lem:
    if i == '':
        temp=''
    else:
        temp=lem_text(i)
    temp_list.append(temp)

df=df.assign(sentence_lower_removePunNum_en_eng_adv_blank_senti_lem=temp_list)
df.to_excel("5479_preprocess_update_04_11.xlsx")





















