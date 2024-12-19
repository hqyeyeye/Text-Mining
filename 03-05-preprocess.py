# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 10:35:13 2024

@author: huqio
"""

import pandas as pd
# read the clean data
df=pd.read_csv("step2_pivotwide/pivot_4000_wide.csv")

# =============================================================================
# get unique visit id for ALL patients , then we have 34828 patients.
# =============================================================================
# make sure no dyplicates
df_uniq=df.medicalvisit_id.unique()

#combine two text comments

df["text_exp"].fillna('',inplace=True)
df["text_provider"].fillna('',inplace=True)

df['exp_pro'] = df[["text_exp","text_provider"]].apply(lambda s: s.str.cat(sep=' '), axis=1)

# only experience
df['exp_pro']=df['text_exp']


# find non empty comments 
# include comments for provider

df1= df[df["exp_pro"].str.len() == 1]
# Merge df and df1 with indicator argument
merged = pd.merge(df, df1, how='outer', indicator=True)

# Filter out rows only present in df
rows_only_in_df = merged[merged['_merge'] == 'left_only'].drop(columns='_merge')
df1=rows_only_in_df

# no comments for provider
df1= df[df["text_exp"].str.len() > 0]

# put all the reviews with comments into a new file
df1.to_excel("pod_5479_exp_04_11.xlsx")




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
df = pd.read_excel("pod_5479_exp_04_11.xlsx")

# just for those pods which no need to combine the comment for the provider and the experience
# df = pd.read_excel("pod_4013_exp.xlsx")
# df=df.rename(columns={"text_exp": "exp_pro"})
# new empty dataframe
new_df = pd.DataFrame(columns=['index', 'comment_index', 'sentence', 'non_stop','medicalvisit_id'])
flag = 0

for i, row in df.iterrows():
    num = i
    row_text=row["exp_pro"]
    cleaned_row = clean_text(str(row_text))
    word_tokens = word_tokenize(cleaned_row)
    end_word = []
    non_stop_text = []
    medicalvisit_id=row["medicalvisit_id"]
    for j in range(len(word_tokens)):
        i = word_tokens[j]
        end_word.append(i)
        if i in ('.', '!', '?', '！', '？','..') or j == len(word_tokens) - 1:
            text = " ".join(end_word)
            if len(text) > 1:
                non_stop_text = [word for word in end_word if word.lower() not in stop_words]
                # replace "nan" as empty string
                non_stop_text = ['' if word.lower() == 'nan' else word for word in non_stop_text]
                if all([re.match(r'\W', word) for word in non_stop_text]):
                    non_stop_text = []
                text = ['' if word.lower() == 'nan' else word for word in end_word]

                new_df = new_df.append({'index': flag, 'comment_index': num, 'sentence': " ".join(text),
                                        'non_stop': " ".join(non_stop_text), 'medicalvisit_id':medicalvisit_id},
                                       ignore_index=True)
                # print(flag, text)
                flag += 1
            end_word.clear()
            non_stop_text.clear()

#save the new df to a excel file only including all texts information            
new_df.to_excel("5479_text_04_11.xlsx")

## easy method to split sentence
# from nltk.tokenize import sent_tokenize
df=pd.read_excel("5479_text_04_11.xlsx")


####### improt packages to translate all the comments to english
from langdetect import detect
from deep_translator import GoogleTranslator


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


























###########################################
# merge all the stopwords and sentiment words
from nltk.tokenize import word_tokenize
# nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

stop_words = stopwords.words('english')
with open('stopwords.txt', encoding="utf8") as f:
    lines = f.readlines()
    adv_stop_words = lines

import pandas as pd
df_st=pd.read_excel("sentiment_words.xlsx")
senti_words=df_st["words"].unique().tolist()


stop_words.extend(x for x in adv_stop_words if x not in stop_words)
stop_words.extend(x for x in senti_words if x not in stop_words)

pd.DataFrame(stop_words).to_excel("all_stopwors_sentiment.xlsx")

# =============================================================================
# Find all the open questions; check how many of them have text infor
#(51504 open questions/27883 has no response/59 left blank/ 23563 open-questions receive comments)
#only 45.75% left commnets for open-questions
# =============================================================================
open_question=df.loc[df["question_iscomment"]==True]
empty=open_question.loc[open_question['scalevalue_reporttext']=='Non-Response']
exclude1=open_question[~open_question.scalevalue_reporttext.isin(empty.scalevalue_reporttext)]
empty1=open_question.loc[open_question['scalevalue_reporttext']=='(BLANK)']
exclude2=exclude1[~exclude1.scalevalue_reporttext.isin(empty1.scalevalue_reporttext)]

# number of patiens who left comments 18405 patients, 18405/ 34828 =52.85%
idx=exclude2.medicalvisit_id.unique()
idx=idx.tolist()
# get full information for those patients who left comments
df_text=df[~df['medicalvisit_id'].isin(idx)]
df_text.to_csv("visits_infor_commentsYes.csv")


# import pandas as pd
# from googletrans import Translator


# Assuming your non-English text is in a column named 'NonEnglishText'
# non_english_column = 'sentence_lower_remPunNum'

# # Initialize translator
# translator = Translator()

# # Translate each non-English text to English
# translated_texts = []
# for text in df[non_english_column]:
#     translation = translator.translate(text, src='auto', dest='en')
#     translated_texts.append(translation.text)

# # Add translated text to DataFrame
# df['TranslatedText'] = translated_texts


# dff=pd.read_excel("4000_preprocess.xlsx")
# # Find differences between two columns
# diff_indices = []
# diff_content = []

# for idx, (val1, val2) in enumerate(zip(dff['sentence_lower_removePunNum_en'], dff['sentence_lower_remPun'])):
#     if val1 != val2:
#         diff_indices.append(idx)
        # diff_content.append((val1, val2))

# Print indices and content where values differ
# for idx, content in zip(diff_indices, diff_content):
#     print(f"Index: {idx}, Content: {content}")





