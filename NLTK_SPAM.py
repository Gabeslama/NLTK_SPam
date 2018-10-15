#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

#nltk.download_shell()

messages = [line.rstrip() for line in open('/Users/gabrielslama/Desktop/smsspamcollection/SMSSpamCollection')]

print(len(messages))


#first 10 messages 

for mess_no,message in enumerate(messages[:10]):
    print(mess_no,message)
    print('\n')  

messages = pd.read_csv('SMSSpamCollection',sep='\t',names=['label','message'])

#Data Analysis and Visualization
messages.groupby('label').describe() 

#How long text messages are
messages['length'] = messages['message'].apply(len)


messages['length'].plot.hist(bins=150)
messages['length'].describe()

#largest text message analysis
messages[messages['length'] == 910]['message'].iloc[0]
messages.hist(column='length',by='label',bins=60)

mess = ('Sample message! Notice: it has punctuation.')
nopunc = [c for c in mess if c not in string.punctuation]

nopunc = ''.join(nopunc)

x = ['a','b','c','d']

''.join(x) #Joins them together


#Remove punctuation
#Remove stopwords
#Returning the list of text words that are cleaned
def text_process(mess):

    nopunc = [char for char in mess if char not in string.punctuation]
    
    #joining elements in list together
    nopunc = ''.join(nopunc)
    
    #return if elements in list not in stopwords (all lowercase)
    return[word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

messages['message'].head(5).apply(text_process)

#Convert each message to vector for ML model to understand 

"""
1.Count number of times of occurance (frequency)
2. Weigh the counts, frequeny tokens get lower weight (inverse document frequency )
3.Normalize vectors to unit length, to abstract from original text length.
"""
#bag of words model
bow_transformer = CountVectorizer(analyzer=text_process).fit(messages['message'])
print(len(bow_transformer.vocabulary_)) #11425

mess4 = messages['message'][3]

bow4= bow_transformer.transform([mess4])
print(bow4)
print(bow4.shape)
bow_transformer.get_feature_names()[9554] #seeing what word shows up twice

messages_bow = bow_transformer.transform(messages['message'])
print('Shape of Sparse Matrix: ', messages_bow.shape)

#Weight value for each word
tfidf_transformer = TfidfTransformer().fit(messages_bow)
tfidf4 = tfidf_transformer.transform(bow4)

messages_tfidf = tfidf_transformer.transform(messages_bow)

#Predict ham vs spam
spam_detect_model = MultinomialNB().fit(messages_tfidf,messages['label'])


#Train test split
msg_train,msg_test,label_train,label_test = train_test_split(messages['message'],
                                                             messages['label'],test_size=0.3)

#Pipeline
pipeline = Pipeline([
        ('bow',CountVectorizer(analyzer=text_process)),
        ('tfidf',TfidfTransformer()),
        ('classifier',MultinomialNB())
        ])
    
pipeline.fit(msg_train,label_train)

predictions = pipeline.predict(msg_test)
print(classification_report(label_test,predictions))

#95% precision 
