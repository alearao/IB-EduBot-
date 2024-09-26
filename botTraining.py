#libraries 
import random #choosing the random intent responses 
import json #to access the intents 
import pickle #serializing 
import numpy as np 
import os
import nltk
#import ssl
#Disable SSL certificate verification, need for setting up nltk 
#import ssl
#ssl._create_default_https_context = ssl._create_unverified_context

from nltk.stem import WordNetLemmatizer #reduces the word to its stem 

#import from tensorflow: available ML and AI software for model training
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD #gradient decent function 

stemmer = WordNetLemmatizer() #creates a word stemmer 
#making sure the AI can recognise the word is easier at its simplest form

intents = json.loads(open('intentsTest.json').read()) #loading and reading json file
#result: getting a json object of the intents file 

wordList = []
tags = [] #tag 
documents = [] 
ignore = ['.' , ',' , '!' , '?' , ':'] #to ignore in input phrases


stemmer = WordNetLemmatizer() #creates a word stemmer 
#making sure the AI can recognise the word is easier at its simplest form

#looping through the intents file 
for intent in intents['intents']:
    for pattern in intent['patterns']:
        words = nltk.word_tokenize(pattern)
        wordList.extend(words)
        documents.append((words, intent['tags']))
        if intent['tags'] not in tags:
            tags.append(intent['tags'])
            
wordList = [stemmer.lemmatize(word) for word in wordList if word not in ignore]

wordList = sorted(set(wordList)) #sorts + lemmatizes
tags = sorted(tags)
#create pickle files containing all words +classes 
pickle.dump(wordList,open('words.pkl','wb'))
pickle.dump(tags,open('tags.pkl','wb')) 



#training: need numerical values to feed into the neural network 
#set individual word values to 0, 1 depending if its occuring in the pattern 
trainingList = []
outputEmpty = [0]*len(tags) #as many 0s as classes 

for document in documents: #for each combination create empty BOW 
    bow = []
    word_patterns = document[0]
    word_patterns = [stemmer.lemmatize(word.lower()) for word in word_patterns]
    #for each word we want to know if it occurs in the patterns 
    for word in wordList: 
       #appending 1 to bag if word occurs in word patterns 
        bow.append(1) if word in word_patterns else bow.append(0)
        outputRow = list(outputEmpty)
        outputRow[tags.index(document[1])]=1 #set to 1 
        trainingList.append([bow, outputRow])
 
random.shuffle(trainingList) #shuffles training list
x_train = [item[0] for item in trainingList] #turns x, y training data into numpy arrays
y_train = [item[1] for item in trainingList]

model = Sequential()
model.add(Dense(128, input_shape=(len(x_train[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(y_train[0]), activation='softmax')) #softmax allows to add up results, add up percentages to near 1 

# Create model - 3 layers. First layer 128 neurons, 
# second layer 64 neurons and 3rd output layer contains number of neurons
# equal to number of intents to predict output intent with softmax        
#Stochastic Gradient Descent (SGD) optimizer class from TensorFlow's Keras API.
sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
mod = model.fit(np.array(x_train), np.array(y_train), epochs = 200, batch_size = 5, verbose = 1)
#feed same data 200 times into neural network, in batches of 5 
model.save('chatbot_model.h5', mod)
print("model created!")

