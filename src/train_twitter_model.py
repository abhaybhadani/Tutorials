#!/usr/bin/env python
# coding: utf-8

# #  Analyzing the sentiments of the people from micro-blogging site / social media sites (Twitter)
# 
# 
# ### Dr. Abhay Bhadani
# #### Sr. Director/Head (Data Science)
# #### Yatra Online Ltd., Gurgaon
# #### Ph.D. (IIT Delhi)
# 
# 

# Emotions are expressed in nuanced ways, which varies by collective or individual experiences, knowledge, and beliefs. Therefore, to understand emotion, as conveyed through text, a robust mechanism capable of capturing and modeling different linguistic nuances and phenomena is needed. Emotions reflect different users’ perspectives towards actions and events, therefore they are innately expressed in dynamic linguistic forms.
# 
# 
# Consider the social posts <B>“Thanks God for everything”</B> and <B>“Tnx mom for waaaaking me two hours early. Cant get asleep now”,</B> a lexicon-based model may not properly represent the emotion-relevant phrases: <I> “waaaaking me”, “Thanks God”, and “Tnx mom”. </I> First, the word “waaaaking” doesn’t exist in the English vocabulary, hence its referent may vary from its standard form, “waking”. Secondly, knowledge of the semantic similarity between the words <B>“Thanks” and “Tnx” </B> is needed to establish any relationship between the last two phrases. Even if such relationship can be established through knowledgebased techniques, it’s difficult to reliably determine the association of these phrases to a group of emotions. 
# 
# 
# <B>Sentiment analysis is part of the Natural Language Processing (NLP).
# 
# <B> It is a type of text mining which aims to determine the opinion and subjectivity of its content. 
#     
# We can extract emotions related to some raw texts (e.g., reviews, comments, tweets). This is usually used on social media posts, customer reviews, customer queries, etc.  
#     
# Every customer facing industry (retail, telecom, finance, etc.) or political party or any such organizations are interested in identifying their customers’ sentiment, whether they think positive or negative, are they happy, sad, and so on about them.
#     
# Today, we shall perform a study to show how sentiment analysis can be performed using Python and how it can be deployed in production systems and host as an API.

# ![sen](./images/performing-twitter-sentiment-analysis1.jpg)

# ###  Dataset:
# 
# <!-- https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp -->
# 
# <B>Description
# 
# The data is in csv format. In computing, a comma-separated values (CSV) file stores tabular data (numbers and text) in plain text. Each line of the file is a data record. Each record consists of one or more fields, normally separated by commas. However, in this case the separator used in a semi-colon.
# 
# 
# Here, we will aggregate Tweets based on sentiment. The aggregation process is based on the association of tweets with the same feelings, as well as the degree and proportion of the feeling.
# 
# The dataset consists of sentences that have been classified into the following categories: {'sadness', 'anger', 'love', 'surprise', 'fear', 'joy'}
#     
#     
# List of documents with emotion flag, Dataset is split into train, test & validation for building the machine learning model
# 
# Example :-
#     
#     i feel like I am still looking at a blank canvas blank pieces of paper;sadness
#     
#     i cant walk into a shop anywhere where i do not feel uncomfortable;fear
#     
#     i felt anger when at the end of a telephone call;anger
# 
#     i never make her separate from me because i don t ever want her to feel like i m ashamed with her;sadness
#     
# 
# The methodology used is based on building a classifier using different algorithms (such as recurrent neural network) that is capable of analyzing sentiment, using a data set that includes a number of emotions.
# 
#      
#     
#  

# ### Data Pre-Processing
# ![pre](https://www.electronicsmedia.info/wp-content/uploads/2017/12/Data-Preprocessing.jpg)

#    
# <B> Approach
#     
# Text Cleaning Steps:
#     
#     1) Clean the data
#         Removing Twitter Handles (@user), Punctuations, Numbers, and Special Characters, Stop Words,
#         Removing Short Words
#     
#     2) Perform Tokenization:  
#             Tokens are individual terms or words, and tokenization is the process of splitting a string of text into tokens.
#     
# ![vis](./images/tokenization.png)
#     
#     3) Stemming:
#             Stemming is a rule-based process of stripping the suffixes (“ing”, “ly”, “es”, “s” etc) from a word. For example, For example – “play”, “player”, “played”, “plays” and “playing” are the different variations of the word – “play”
#     
# ![vis](./images/stemming.jpeg)
#     
#     4) Visualization the Tweets using WordCloud:
#             A wordcloud is a visualization wherein the most frequent words appear in large size and the less frequent words appear in smaller sizes.
# 
# 
# ![vis](./images/word-cloud-sample.png)
#     
# 
# The next stage involves using the trained model to sort tweets based on sentiment with a rating ratio.
#     
# 

# In this partial stage, we will follow two methodologies: 
#     
#     The first is to draw a graph that shows the percentage of each of the feelings of the tweeters within Twitter regarding what is happening in the state of Sri Lanka.
#     
#     The next partial stage, is to move to the study of each of these feelings for the tweeters, and try to collect them in order to determine the degree of feelings for each of them.
#     
#     The final hierarchical schemas (for each one of the feelings) will show the correlation of the tweeters in terms of the degree of affiliation with that feeling.
#     
# The Euclidean distance will be used to calculate the degree of convergence for a single feeling (depending on the percentage of tweeting classification and belonging to a specific feeling).

# ### Representation of Words as Vectors
# 
# There are various ways to represent words in Vector Format.
#     Bag-Of-Words
#     
#     Term Frequecy - Inverse Document Frequecy (TF-IDF)
#     
#     Word2Vec (Skip-Gram and CBOW)
#     
#     GloVe: GloVe stands for Global Vectors for word representation.
#   
#    
#     Fast-Text: FastText was introduced by Facebook back in 2016. The idea behind FastText is very similar to Word2Vec. However, there was still one thing that methods like Word2Vec and GloVe lacked. Even though both of these models have been trained on billions of words, that still means our vocabulary is limited. FastText improved over other methods because of its capability of generalization to unknown words, which had been missing all along in the other methods.
#     
#     Bidirectional Encoder Representations from Transformers (BERT): BERT is a transformer-based architecture. Transformer uses a self-attention mechanism, which is suitable for language understanding. BERT is a multi-layered encoder. 
#         
#         BERT base – 12 layers, 12 attention heads, and 110 million parameters. 
#         
#         BERT Large – 24 layers, 16 attention heads and, 340 million parameters.
# 
# 
#     
#     Sentence - Encoders
# 
# 
# <B>Vectorization</B> is jargon for a classic approach of converting input data from its raw format (i.e. text ) into vectors of real numbers which is the format that ML models support. This approach has been there ever since computers were first built, it has worked wonderfully across various domains, and it’s now used in NLP.
# 
# 
# Download a pretrained vector representation of the words. 
# These pre-trained vectors have been trained using GloVe embedding technique.

# ![viz](./images/GloVe_Representation.png)

# In[ ]:





# In[ ]:


# !wget https://nlp.stanford.edu/data/glove.6B.zip


# In[ ]:


# !unzip glove.6B.zip


# In[ ]:


# import opendatasets as op


# In[ ]:


# dataset_emotion = "emotions-dataset-for-nlp"


# #### Install and import relevant python Packages:

# In[ ]:


# !pip install sklearn seaborn matplotlib tensorflow keras nltk flask requests 


# In[ ]:


import pandas as pd
import os
import numpy as np
import tensorflow as tf
import keras
import nltk
import string
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Embedding, LSTM ,Conv2D, Dense,GlobalAveragePooling1D,Flatten, Dropout , GRU, TimeDistributed, Conv1D, MaxPool1D, MaxPool2D
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
import matplotlib as mpl


# ### First Step:
# building a recurrent neural network capable of analyzing emotions, using a dataset that includes a number of emotions.

# In[ ]:


from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
nltk.download('stopwords')
porter = PorterStemmer()
stop_words = stopwords.words('english')


# In[ ]:


class Emotion:
  def __init__(self, datasetFolder, batch_size, validation_split, optimizer, loss, epochs):
    self.datasetFolder = datasetFolder
    self.batch_size = batch_size
    self.validation_split = validation_split
    self.optimizer = optimizer
    self.loss = loss
    self.epochs = epochs
  def readDatasetCSV(self):
    trainDataset = pd.read_csv(os.path.join(self.datasetFolder, "data/train.txt"), names=['Text', 'Emotion'], sep=';')
    testDataset = pd.read_csv(os.path.join(self.datasetFolder, "data/test.txt"), names=['Text', 'Emotion'], sep=';')
    validDataset = pd.read_csv(os.path.join(self.datasetFolder, "data/val.txt"), names=['Text', 'Emotion'], sep=';')
    list_dataset = [trainDataset, testDataset, validDataset]
    self.dataset = pd.concat(list_dataset)
  def FeaturesLables(self):
    self.features = self.dataset['Text']
    self.labels = self.dataset['Emotion']  
  def splitDataset(self):
    self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.features,
                                                                            self.labels, 
                                                                            test_size = self.validation_split)
  def CleanFeatures(self):
    self.features = self.features.apply(lambda sequence:
                                              [ltrs.lower() for ltrs in sequence if ltrs not in string.punctuation])
    self.features = self.features.apply(lambda wrd: ''.join(wrd))
  def tokenizerDataset(self):
    self.tokenizer = Tokenizer(num_words=5000)
    self.tokenizer.fit_on_texts(self.features)
    train = self.tokenizer.texts_to_sequences(self.features)
    self.features = pad_sequences(train)
    le = LabelEncoder()
    self.labels = le.fit_transform(self.labels)
    self.vocabulary = len(self.tokenizer.word_index)
  def label_categorical(self):
    self.labels = to_categorical(self.labels, 6)
  def glove_word_embedding(self, file_name):
    self.embeddings_index = {}
    file_ = open(file_name)
    for line in file_:
        arr = line.split()
        single_word = arr[0]
        w = np.asarray(arr[1:],dtype='float32')
        self.embeddings_index[single_word] = w
    file_.close()
    max_words = self.vocabulary + 1
    word_index = self.tokenizer.word_index
    self.embedding_matrix = np.zeros((max_words,300)).astype(object)
    for word , i in word_index.items():
            embedding_vector = self.embeddings_index.get(word)
            if embedding_vector is not None:
                self.embedding_matrix[i] = embedding_vector  
  def model(self):
    m = Sequential()
    m.add(Input(shape=(self.features.shape[1], )))
    m.add(Embedding(self.vocabulary + 1,300))
    m.add(GRU(128, recurrent_dropout=0.3, return_sequences=False, activity_regularizer = tf.keras.regularizers.L2(0.0001)))
    m.add(Dense(6, activation="softmax", activity_regularizer = tf.keras.regularizers.L2(0.0001)))
    self.m = m
  def compiler(self):
    self.m.compile(loss= self.loss,optimizer=self.optimizer,metrics=['accuracy'])
  def fit(self):
    earlyStopping = EarlyStopping(monitor = 'loss', patience = 20, mode = 'min', restore_best_weights = True)
    self.history_training = self.m.fit(self.X_train, self.Y_train, epochs= self.epochs,batch_size = self.batch_size,
                                       callbacks=[ earlyStopping])   
    
  def save_model(self, model_file='./models/model.json'):
    # serialize model to JSON
    model_json = self.m.to_json()
    with open(model_file, "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
        self.m.save_weights(model_file+".h5")
    print("Saved model to disk")


# In[ ]:


dataset_emotion = "."
epochs = 1
emotion = Emotion(dataset_emotion, 256, 0.1, 'adam', 'categorical_crossentropy', epochs)


# In[ ]:


emotion.readDatasetCSV()


# In[ ]:


emotion.dataset.head()


# In[ ]:


emotion.FeaturesLables()


# In[ ]:


emotion.CleanFeatures()


# In[ ]:


emotion.features.head()


# In[ ]:


emotion.labels.unique()


# In[ ]:


emotion.tokenizerDataset()


# In[ ]:


emotion.features


# In[ ]:


emotion.labels


# In[ ]:


emotion.features.shape


# In[ ]:


emotion.features.shape


# In[ ]:


emotion.label_categorical()


# In[ ]:


emotion.labels


# In[ ]:


emotion.splitDataset()


# In[ ]:


emotion.glove_word_embedding("./pre-trained-embeddings/glove.6B.300d.txt")


# In[ ]:


emotion.model()
emotion.m.layers[0].set_weights([emotion.embedding_matrix])
emotion.m.layers[0].trainable = False


# In[ ]:


emotion.compiler()


# In[ ]:


emotion.m.summary()


# In[ ]:


emotion.fit()


# In[ ]:


emotion.save_model('./models/test.json')
 


# In[ ]:





# In[ ]:


import matplotlib.pyplot as plt
mpl.style.use('seaborn')
figure = plt.figure(figsize=(15, 4))
plt.plot(emotion.history_training.history['accuracy'], 'darkorange', label = 'Accuracy')
plt.title("Accuracywhile training")
plt.show()


# In[ ]:


figure = plt.figure(figsize=(15, 4))
plt.plot(emotion.history_training.history['loss'], 'darkblue', label = 'Loss')
plt.title("Loss while training")
plt.show()


# In[ ]:


emotion.m.evaluate(emotion.X_test, emotion.Y_test, batch_size = 256)


# In[ ]:


y_pred = emotion.m.predict(emotion.X_test)


# In[ ]:


y_pred = np.argmax(y_pred, axis = 1)


# In[ ]:


y_pred


# In[ ]:


y_test = np.argmax(emotion.Y_test, axis = 1)


# In[ ]:


y_test


# In[ ]:


from sklearn.metrics import accuracy_score as acc
print(acc(y_pred, y_test))


# In[ ]:


res = tf.math.confusion_matrix(y_pred,y_test).numpy()


# In[ ]:


cm = pd.DataFrame(res,
                     index = ['sadness', 'anger', 'love', 'surprise', 'fear', 'joy'], 
                     columns = ['sadness', 'anger', 'love', 'surprise', 'fear', 'joy'])
cm


# In[ ]:





# In[ ]:


import seaborn as sns
figure = plt.figure(figsize=(7, 7))
sns.heatmap(cm, annot=True, cmap=plt.cm.Blues)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()


# In[ ]:





# ### Second Step:
# Using the model that has been trained to sort tweets based on sentiment with a rating ratio.

# In[ ]:


SriLankaTweets = "./data/SriLankaTweets.csv"


# In[ ]:


SriLankaTweets = pd.read_csv(SriLankaTweets)


# In[ ]:


SriLankaTweets.head()


# In[ ]:


SriLankaTweets.describe()


# In[ ]:


SriLankaTweets['language'].unique()


# #### Dataset pretreatment

# In[ ]:


SriLankaTweets = SriLankaTweets.loc[SriLankaTweets['language'] == 'en']


# In[ ]:


len(SriLankaTweets)


# In[ ]:


SriLankaTweets['tweet'].dropna()


# In[ ]:


def preprocessingText(sentences):
  sentences = sentences.apply(lambda sequence:
                                              [ltrs.lower() for ltrs in sequence if ltrs not in string.punctuation])
  sentences = sentences.apply(lambda wrd: ''.join(wrd))
  return sentences


# In[ ]:


type(SriLankaTweets['tweet'][5])


# In[ ]:


SriLankaTweets['tweet'] = preprocessingText(SriLankaTweets['tweet'])


# In[ ]:


SriLankaTweets.head()


# In[ ]:





# In[ ]:


features = SriLankaTweets['tweet']


# In[ ]:


features.head()


# In[ ]:


len(features)


# In[ ]:


features.shape


# In[ ]:


tweets = emotion.tokenizer.texts_to_sequences(features)
tweets = np.array(tweets).reshape(-1)
tweets = pad_sequences(tweets, maxlen= 63)


# In[ ]:


tweets


# In[ ]:





# ##Using the sentiment analysis model:
# Using the trained sentiment analysis model, in order to analyze the sentiments of tweeters within the Sri Lanka dataset.
# Sentiment type and sentiment affiliation will be preserved for each Tweet.

# In[ ]:


sentiment_labels = ['sadness', 'anger', 'love', 'surprise', 'fear', 'joy']
sentiment_labels_encoding = [0, 1, 2, 3, 4, 5]


# In[ ]:


len(tweets)


# In[ ]:


results_sen_tweets = emotion.m.predict(tweets, batch_size = 256)


# In[ ]:


len(results_sen_tweets)


# In[ ]:


sentiments = []
sentiment_labels1=[]
sentiment_labels2=[]

for i in results_sen_tweets:
  res = np.argmax(i, axis = 0)
  sentiments.append([sentiment_labels_encoding[res], i[res]])
  sentiment_labels1.append(sentiment_labels_encoding[res])
  sentiment_labels2.append(sentiment_labels[res])


# In[ ]:


data= { "tweets":SriLankaTweets['tweet'],
       "labels_num": sentiment_labels1, 
       "labels_text": sentiment_labels2 

      }

tweet_labels_df = pd.DataFrame(data)
tweet_labels_df.head()


# In[ ]:





# In[ ]:





# ### A graph showing the distribution of tweeters' feelings regarding events in Sri Lanka

# In[ ]:


def count_sent(sentiments, depending_on):
  c = 0
  for i in sentiments:
    if i[0] == depending_on:
      c = c + 1
  return c    


# In[ ]:


arr = []
for i in sentiment_labels_encoding:
  arr.append(count_sent(sentiments, i))


# In[ ]:


arr


# In[ ]:


plt.pie(arr, labels = sentiment_labels)
plt.show()


# ####  The hierarchical distribution of each feeling:
# This stage aims to determine the degree of convergence in terms of the single feeling of the tweeters, depending on the aggregation process based on the Euclidean distance, which depends on the percentage of feeling classification.

# In[ ]:


tweet_labels_df['labels_num'].unique()


# In[ ]:


from wordcloud import WordCloud,ImageColorGenerator
from PIL import Image
import urllib
import requests


# ### Filter out the words of a class i.e. ('sadness', 'anger', 'love', 'surprise', 'fear', 'joy')

# In[ ]:





# In[ ]:


apply_filter ='sadness'

filtered_words = ' '.join(text for text in tweet_labels_df['tweets'][tweet_labels_df['labels_text']==apply_filter])


# In[ ]:


# combining the image with the dataset
# Mask = np.array(Image.open(requests.get('http://clipart-library.com/image_gallery2/Twitter-PNG-Image.png', stream=True).raw))
Mask = np.array(Image.open('./images/Twitter-PNG-Image.png'))

# We use the ImageColorGenerator library from Wordcloud 
# Here we take the color of the image and impose it over our wordcloud
image_colors = ImageColorGenerator(Mask)

# Now we use the WordCloud function from the wordcloud library 
wc = WordCloud(background_color='black', height=1500, width=4000,mask=Mask).generate(filtered_words)

# Size of the image generated 
plt.figure(figsize=(10,20))

# Here we recolor the words from the dataset to the image's color
# recolor just recolors the default colors to the image's blue color
# interpolation is used to smooth the image generated 
plt.imshow(wc.recolor(color_func=image_colors),interpolation="hamming")

plt.axis('off')
plt.show()


# In[ ]:





# # Extracting Features from cleaned Tweets

# ### Bag-of-Words Features

# Bag of Words is a method to extract features from text documents. These features can be used for training machine learning algorithms. It creates a vocabulary of all the unique words occurring in all the documents in the training set. 
# 
# Consider a corpus (a collection of texts) called C of D documents {d1,d2…..dD} and N unique tokens extracted out of the corpus C. The N tokens (words) will form a list, and the size of the bag-of-words matrix M will be given by D X N. Each row in the matrix M contains the frequency of tokens in document D(i).
# 
# For example, if you have 2 documents-
# 
# 
# 
# - D1: He is a lazy boy. She is also lazy.
# 
# - D2: Smith is a lazy person.
# 
# First, it creates a vocabulary using unique words from all the documents
# #### [‘He’ , ’She’ , ’lazy’ , 'boy’ ,  'Smith’  , ’person’] 
# 
# - Here, D=2, N=6
# 
# 
# 
# - The matrix M of size 2 X 6 will be represented as:
# 
# ![bow](https://s3-ap-south-1.amazonaws.com/av-blog-media/wp-content/uploads/2018/07/table.png)
# 
# The above table depicts the training features containing term frequencies of each word in each document. This is called bag-of-words approach since the number of occurrence and not sequence or order of words matters in this approach.

# ### TF-IDF Features

# Tf-idf stands for term frequency-inverse document frequency, and the tf-idf weight is a weight often used in information retrieval and text mining. This weight is a statistical measure used to evaluate how important a word is to a document in a collection or corpus. The importance increases proportionally to the number of times a word appears in the document but is offset by the frequency of the word in the corpus. 
# 
# Typically, the tf-idf weight is composed by two terms: the first computes the normalized Term Frequency (TF), aka. the number of times a word appears in a document, divided by the total number of words in that document; the second term is the Inverse Document Frequency (IDF), computed as the logarithm of the number of the documents in the corpus divided by the number of documents where the specific term appears.
# 
# - TF: Term Frequency, which measures how frequently a term occurs in a document. Since every document is different in length, it is possible that a term would appear much more times in long documents than shorter ones. Thus, the term frequency is often divided by the document length (aka. the total number of terms in the document) as a way of normalization: 
# #### TF(t) = (Number of times term t appears in a document) / (Total number of terms in the document).
# 
# - IDF: Inverse Document Frequency, which measures how important a term is. While computing TF, all terms are considered equally important. However it is known that certain terms, such as "is", "of", and "that", may appear a lot of times but have little importance. Thus we need to weigh down the frequent terms while scale up the rare ones, by computing the following: 
# #### IDF(t) = log_e(Total number of documents / Number of documents with term t in it).
# 
# #### Example:
# 
# Consider a document containing 100 words wherein the word cat appears 3 times. The term frequency (i.e., tf) for cat is then (3 / 100) = 0.03. Now, assume we have 10 million documents and the word cat appears in one thousand of these. Then, the inverse document frequency (i.e., idf) is calculated as log(10,000,000 / 1,000) = 4. Thus, the Tf-idf weight is the product of these quantities: 0.03 * 4 = 0.12.
# 
# 

# In[ ]:






# ### Deploy the Classifier as an API

# In[2]:


from flask import Flask
import numpy as np
from keras.models import model_from_json
from flask import jsonify, request
import json
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import pandas as pd
import string
# import requests

app = Flask(__name__)

# load json and create model
json_file = open('./models/test.json', 'r')
loaded_model = model_from_json(json_file.read())
json_file.close()

# load weights into new model
loaded_model.load_weights("./models/test.json.h5")
print("Loaded model from disk")

sentiment_labels = ['sadness', 'anger', 'love', 'surprise', 'fear', 'joy']
sentiment_labels_encoding = [0, 1, 2, 3, 4, 5]


def preprocessingText1(sentences):
    sentences = sentences.apply(lambda sequence:
                                              [ltrs.lower() for ltrs in sequence if ltrs not in string.punctuation])
    sentences = sentences.apply(lambda wrd: ''.join(wrd))
    return sentences


@app.route("/predict_sentiment", methods=['POST'])
def predict_sentiment(text):
    try:
        params = json.loads(request.get_data())
        text = params.get("query",text)
    except Exception as e:
        print('text= ', text)

    tokenizer = Tokenizer(num_words=5000)
    tw = preprocessingText1(pd.Series(text))
    tw = tokenizer.texts_to_sequences(tw)
    tw = pad_sequences(tw, maxlen= 63)
    prob =loaded_model.predict(tw)
    idx=pd.Series(prob[0]).idxmax()
    return {'Label': sentiment_labels[idx], 'model probability score':prob}
#     return jsonify({'Label': sentiment_labels[idx], 
#                     'model probability score':prob})


# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=105)

test_sentence1 = "I enjoyed my journey on this flight."
print(predict_sentiment(test_sentence1))


