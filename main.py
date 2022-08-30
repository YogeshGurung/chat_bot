# robo-chatbot
import io
from operator import index
from posixpath import split
import random
from secrets import choice
import string
import token
import warnings
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings("ignore")

import nltk
from nltk.stem import WordNetLemmatizer
nltk.download("popular", quiet=True)

#read the text file
with open("ganesha.txt","r", encoding="utf8", errors="ignore") as words:
    raw_data = words.read().lower()

#Tokenization
#converts raw data to sentences
token_sentence = nltk.sent_tokenize(raw_data)
#converts raw data to words
token_word = nltk.sent_tokenize(raw_data)

#Preprocessing the data inputs
lem_processor = WordNetLemmatizer()
#
def lemmerToken(tokens):
    return [lem_processor.lemmatize(token) for token in tokens]
#
dict_punc = dict((ord(punct), None) for punct in string.punctuation)
#
def lemmerNormalize(text):
    return lemmerToken(nltk.word_tokenize(text.lower().translate(dict_punc)))

# keyword match
greetings_inputs = ("namaste","namaskar","hi", "hello", "howdy","hey", "g'day")
greetings_responses = ["namaskar","namaste","hi", "hello", "hey", "hi there"]

# greeting generation
def ganesha_greeting(sentence):
    for word in sentence.split():
        if word.lower() in greetings_inputs:
            return random.choice(greetings_responses)

# generate response to user
def ganesha_response(userResponse):
    ganeshaResponse = " "
    token_sentence.append(userResponse)
    Tfid_fVectorizer = TfidfVectorizer(tokenizer=lemmerNormalize, stop_words="english")
    Tfid = Tfid_fVectorizer.fit_transform(token_sentence)
    cos_val = cosine_similarity(Tfid[-1],Tfid)
    index_val = cos_val.argsort()[0][-2]
    flat = cos_val.flatten()
    flat.sort()
    req_Tfid = flat[-2]
    if (req_Tfid==0):
        ganeshaResponse=ganeshaResponse+"Could you please rephrase what you mean?"
        return ganeshaResponse
    else:
        ganeshaResponse = ganeshaResponse+ token_sentence[index_val]
        return ganeshaResponse


keepGoing = True
print("Ganesh: Namaste, this is Ganesh. Please ask me your questions. If not, please type Bye!")
#
while(keepGoing == True):
    userResponse = input()
    userResponse = userResponse.lower()
    if (userResponse != "bye"):
        if (userResponse == "thanks" or userResponse == "thank you"):
            keepGoing = False
            print("Ganesh: You are must welcome!")
        else:
            if (ganesha_greeting(userResponse) != None):
                print("Ganesh: "+ganesha_greeting(userResponse))
            else:
                print("Ganesh: ",end="")
                print(ganesha_response(userResponse))
                token_sentence.remove(userResponse)

    else:
        keepGoing = False
        print("Ganesh: See you, please take care of you. Have a good one!")
