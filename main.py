import requests
import streamlit as st
import pandas as pd
import numpy as np
# from streamlit_chat import message
import json
import tflearn
# import tensorflow as tf
header = st.container()
dataset = st.container()
features = st.container()
modelTraining = st.container()
import random
import json

# import pickle
import nltk
nltk.download('punkt')
from nltk.stem.lancaster import  LancasterStemmer
stemmer = LancasterStemmer()
run_bot = True
ERROR_THRESHOLD = 0.25
with open("GL Bot.json") as file:
    corpus = json.load(file)
    data = corpus
corp1 = corpus["intents"]
corp2 = [
    {'tag': 'options',
     'patterns': ["How can you help me", "What can you do", "What help you provide", "What do you know"],
     'responses': ["I can guide you with machine learning and artificial learning concepts", "I can assist you with your program"],
     'context_set': ''
    },
    {'tag': 'SL',
     'patterns': ['bias-variance tradeoff', 'support vector machines', 'normalization', 'Can you help me understand svm',
                  'could you please explain me how machine learning works', 'K nearest neighbor algorithm'
                  'Could me help me understand naive bayes', 'decision trees', 'labeled data', 'supervised', 'Do you know KNN',
                  'Can you help me with KNN algorithm', 'Can you explain KNN', 'Can you help me with supervised ML?'],
     'responses': ['Link: https://en.wikipedia.org/wiki/Supervised_learning'],
     'context_set': ''
    },
    {'tag': 'USL',
     'patterns': ['PCA', 'Pricipal component analysis', 'K means clustering', 'dimensionality reduction', 'unlabeled data'
                  'clustering', 'K nearest neighbor algorithm', 'clustering', 'association', 'hierarchical clustering'
                  'Singular value decomposition', 'Do you know PCA'],
     'responses': ['Link: https://en.wikipedia.org/wiki/Unsupervised_learning'],
     'context_set': ''
    },
    {'tag': 'NN',
     'patterns': ['RelU', 'sigmoid function', 'tanh', 'optimizer', 'neural networks', 'Can you help me with deep learning',
                  'ANN', 'CNN', 'Transfer learning'],
     'responses': ['Link: https://en.wikipedia.org/wiki/Artificial_neural_network'],
     'context_set': ''
    },
    {'tag': 'Bot',
     'patterns': ['May I know who you are', "whats you're name"],
     'responses': ["I'm GL-Bot, your virtual assistant and I'm 24x7 available"],
     'context_set': ''
    },
    {'tag': 'no answer',
     'patterns': [],
     'responses': ["Sorry, I can't understand you", "Please give me more info", "Unable to understand you, Can you provide info"],
     'context_set': ''
    },
    {'tag': 'Intro',
     'patterns': ['How do you do', 'Hi.. Can you help me with my issue', 'Are you there', "what's up", "Howdy",
                  'Could you please help me', 'I need you to listen to me', 'I belong to a new batch of',
                  'Artificial Intelligence class', 'AI & ML batch', 'Machine Learning batch', 'My Program Manager is'],
     'responses': ['Hi there, how can I help you', 'Good to see you again', 'Hello thanks for asking',
                   'Hello, How may I help you'],
     'context_set': ''
    },
    {'tag': 'Exit',
     'patterns': ["that was very helpful", 'thanks again', 'awesome, thanks', 'thanks for helping me out',
                  'will catch you later', "i'll take a leave", 'have a Great day', 'you helped a lot',
                  'you are a great learning buddy', 'bye', 'good bye'],
     'responses': ['Anytime!!', "Happy to help!", "My pleasure!", "Glad I could assist you"],
     'context_set': ''
    },
    {'tag': 'Ticket',
     'patterns': ['my issue is not solved', 'you answer did not help', 'I did not get the answer', 'solution was not good',
                  'you did not help', 'I want to talk to a my PM', 'connect me to the program manager',
                  'I want to raise a ticket', 'no'],
     'responses': ['Transferring the request to your PM', 'Please wait while I transfer your request to PM'],
     'context_set': ''
    }
    ]

for elm2 in corp2:

    for elm1 in corp1:
        if elm2['tag'] == elm1['tag']:
            elm1['responses'].extend(elm2['responses'])
            break
    else:
        corp1.append(elm2)
corpus['intents'] = corp1

words = []
classes = []
documents = []
responses = []
ignore_words = ['?']
for intent in corpus['intents']:
  for pattern in intent['patterns']:
    #tokenize each word in the sentence
    w = nltk.word_tokenize(pattern)
    words.extend(w)
    documents.append((intent['tag'],w))
    if intent['tag'] not in classes:
      classes.append(intent['tag'])
      responses.append(intent['responses'])

# stemmer = LancasterStemmer()
words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
#remove duplicates
classes = sorted(list(set(classes)))
training_data = []
output = []
output_empty = [0]*len(classes)
for doc in documents:
  bag = []
  #List tokenized words for the pattern
  pattern_words = doc[1]
  #stem each word
  pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
  #create bad of words
  for w in words:
      if w in pattern_words:
          bag.append(1)
      else:
          bag.append(0)
    # bag.append(1) if w in pattern_words else bag.append(0)

  #output is a zero for each tag and one for current tag
  output_row = list(output_empty)
  output_row[classes.index(doc[0])] = 1
  training_data.append([bag,output_row])

random.shuffle(training_data)
training_data = np.array(training_data)
train_X = list(training_data[:,0])
train_y = list(training_data[:,1])

net = tflearn.input_data(shape=[None,len(train_X[0])])
net = tflearn.fully_connected(net,8,activation='relu')
net = tflearn.fully_connected(net,8,activation='relu')
net = tflearn.fully_connected(net,len(train_y[0]),activation='softmax')
net = tflearn.regression(net)

model = tflearn.DNN(net,tensorboard_dir='tflearn_logs')
try:
    # model.fit(train_X,train_y,n_epoch = 1000,batch_size=8,show_metric=True)
    # model.save("model.tflearn")
    model.load("model.tflearn")
except:
    model.fit(train_X,train_y,n_epoch = 1000,batch_size=5,show_metric=True)
    model.save("model.tflearn")

def clean_up_sentence(sentence):
  #Breaking sentence into parts
  sentence_words = nltk.word_tokenize(sentence)
  sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
  return sentence_words

def bow(sentence,words,show_details=False):
  sentence_words = clean_up_sentence(sentence)
  bag = [0]*len(words)
  for s in sentence_words:
    for i,w in enumerate(words):
      if w == s:
        bag[i] = 1
  return np.array(bag)

#Method to determine the probability of the words in the sentence given by the user
def classify(sentence):
  results = model.predict([bow(sentence,words)])[0]
  results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD]
  results.sort(key=lambda x:x[1],reverse=True)
  return_list = []
  for r in results:
    return_list.append((classes[r[0]],r[1]))
  return return_list

def response(sentence,show_details=True):
  results= classify(sentence)
  if results:
    while results:
      for i in corpus['intents']:
        if i['tag'] == results[0][0]:
            return random.choice(i['responses'])
          # return print(f"\033[1mBOT-{random.choice(i['responses'])}")
          # return print(random.choice(i['responses']))
      results.pop(0)

def predict_class(sentence):
  res = model.predict([bow(sentence,words)])[0]
  results = [[i,r] for i,r in enumerate(res) if r > ERROR_THRESHOLD]
  results.sort(key=lambda x:x[1], reverse = True)
  return_list = []
  for r in results:
    return_list.append({'intent':classes[r[0]],'probability':str(r[1])})
  return return_list

def start_chat():
  done = False
  print('GO! Bot is Running!!!')
  while not done:
    message = input("\033You- ")
    ints = predict_class(message)
    if ints[0]['intent'] == 'Exit':
      response(message)
      done=True

    else:
      if message.lower()=='' or message.lower()=='*':
        print('Please re-phrase your query!')
        print("\033[1m-"*50)
      elif float(ints[0]['probability']) > 0.35:
        response(message)
        print("\033[1m-"*50)
      else:
        print('Sorry I am still learning!Please ask again!!!')
        print("\033[1m-"*50)


# def chat(user_input):
#     done=False
#     while not done:
#         message = user_input
#         ints = predict_class((message))
#         if ints[0]['intent'] == 'Exit':
#             response(message)
#             done=True
#         else:
#             if message.lower()=='' or message.lower() == '*':
#                 return 'Please re-phrase your query'
#             elif float(int[0]['probability']) > 0.35:
#                 return response(message)
#             else:
#                 return('Sorry I am still learning! Please Rephrase your query')
st.sidebar.title(' GL - BOT')
# st.title("""
# NLP Bot
# NLP Bot is an NLP conversational chatterbot. Initialize the bot by checking the "Initialize bot" radio button.
# """)
run_bot =  st.sidebar.radio('Initialize Bot',('Yes','No'),index=1)
def chat(user_input):
    while True:
        message = user_input
        ints = predict_class((message))
        if ints[0]['intent'] == 'Exit':
            responses = response(message)
            # break
        else:
            responses = response(message)
        return responses
with header:
    st.title('Welcome to Greatlearning CHAT-BOT!!!')
    # st.text('Please Enter your query, The bot will respond')
with features:
    st.header('Chat-Bot Available...You can type your query!!!')

def get_text():
    input_text = st.text_input("You: ","So, What's in your mind")
    return input_text





user_input = get_text()
if True:
    if run_bot == 'Yes':
        st.text_area("Bot: ",value=chat(user_input))
    else:
        st.text_area("Bot:", value="Please start the bot by clicking sidebar radio button", height=200, max_chars=None, key=None)
else:
    st.text_area("Bot:", value="Please start the bot by clicking sidebar radio button", height=200, max_chars=None, key=None)
