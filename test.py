import random
import json
import pickle
import numpy as np
import nltk 
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model

stemmer = WordNetLemmatizer()

intents = json.loads(open('intents.json').read())

words = pickle.load(open('words.pkl','rb'))
tags = pickle.load(open('tags.pkl','rb'))
model = load_model('chatbot_model.h5')

#function for cleaning up the sentence

def clean(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [stemmer.lemmatize(w) for w in sentence_words]
    return sentence_words 

#bag of words function: converting sentence --> bow 
def createBOW(sentence):
    sentence_words = clean(sentence)
    bow = [0]*len(words) #for as many words 
    for w in sentence_words: 
        for i, word in enumerate(words): 
            if word == w: 
                bow[i]=1
    return np.array(bow)

def predict(sentence): 
    bag = createBOW(sentence)
    result = model.predict(np.array([bag]))[0]
    ERROR_THRESHOLD = 0.25 
    res = [[i,r] for i, r in enumerate(result) if r > ERROR_THRESHOLD]
    #predict result, enumerate results to get class, and probability, sort by probability in reverse order
    res.sort(key=lambda x: x[1], reverse = True)
    returnList = []
    for r in res: 
        returnList.append({'intent': tags[r[0]], 'probability': str(r[1])})
    return returnList #return list of intents + 

def getResponse(intents_list, intents_json):
    if intents_list:
        tag = intents_list[0]['intent']
        list_of_intents = intents_json['intents']
        for i in list_of_intents:
            if i['tags'] == tag:
                result = random.choice(i['responses'])
                return {'response': result}
    return {'response': "I apologize, but I'm not sure how to respond to that."}

print("IB TutorBot:")

while True:
    message = input("")
    ints = predict(message)
    print(ints)  # Print the intents list to check its content
    if not ints:
        print("Intents list is empty.")
        print("I apologize, but I'm not sure how to respond to that.")
    else:
        result = getResponse(ints, intents)
        print(result['response'])
