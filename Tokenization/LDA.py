import re
import numpy as np
import pandas as pd
from pprint import pprint

import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

import pyLDAvis
import ast
import main
import warnings
warnings.filterwarnings('ignore')
"""import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt"""


import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

def read_Dataframe(cleandata):
	df = pd.read_csv(cleandata)
	return df

def createDictonary(text):
	id2word = corpora.Dictionary(text)
	return id2word

def copyText(text):
	return text

def tfIdf(texts,id2word):
	corpus = [id2word.doc2bow(text) for text in texts]
	return corpus

def createLDA(corpus,id2word,num_topics):
	lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=num_topics, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=7,
                                           alpha='auto',
                                           per_word_topics=True)

	return lda_model

def displayTopics(lda_model):
	pprint(lda_model.print_topics())

def makelist(text):
	return str(text).split(' ')

def saveLDA(lda_model):
	lda_model.save('../Models/lda_train.model')

def bigrams(words, bi_min=15, tri_min=10):
    bigram = gensim.models.Phrases(words, min_count = bi_min)
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    return bigram_mod

def makenewdf(dataframe):
	#text = dataframe['processed_text'].tolist()

	dataframe['processed_text'] = dataframe['text'].apply(lambda x : makelist(x))
	new_df = dataframe[['stars','processed_text']]
	return new_df

def asliteral(dataframe):
	text = []

	for index,row in dataframe.iterrows():
		res = ast.literal_eval(row['processed_text'])
		text.append(res)

	return text

def create_vecs(lda_train,df,train_corpus):
	train_vecs = []
	
	for i in range(len(df)):
	    top_topics = (
	        lda_train.get_document_topics(train_corpus[i],
	                                      minimum_probability=0.0))
	    topic_vec = [top_topics[i][1] for i in range(15)]
	    topic_vec.extend([len(df.iloc[i].processed_text)])
	    train_vecs.append(topic_vec)
	return topic_vec,train_vecs


def fit(train_vecs,df):

	X = np.array(train_vecs)
	y = np.array(df.stars)

	kf = KFold(5, shuffle=True, random_state=42)

	for train_ind, val_ind in kf.split(X, y):
	    
	    # Assign CV IDX
	    X_train, y_train = X[train_ind], y[train_ind]
	    X_val, y_val = X[val_ind], y[val_ind]
	    
	    # Scale Data
	    scaler = StandardScaler()
	    X_train_scale = scaler.fit_transform(X_train)
	    X_val_scale = scaler.transform(X_val)

	    # Logisitic Regression
	    lr = LogisticRegression(
	        class_weight= 'balanced',
	        solver='newton-cg',
	        fit_intercept=True
	    ).fit(X_train_scale, y_train)

	    y_pred = sgd.predict(X_val_scale)



def pipeline(dataframe):
	
	
	#makenewdf(dataframe)

	dataframe = makenewdf(dataframe)

	text = dataframe['processed_text'].tolist()

	bigram_mod = bigrams(text)
	bigram = [bigram_mod[review] for review in text]

	pprint(bigram[0])
    

	pprint("List of reviews obtained.")

	id2word = createDictonary(bigram)

	print("Created id2word ..")


	corpus = tfIdf(bigram,id2word)

	print("Created corpus...")

	lda_model = createLDA(corpus,id2word,15)
	
	print("LDA complete")

	#saveLDA(lda_model)

	displayTopics(lda_model)

	topic_vec, train_vec = create_vecs(lda_model,dataframe,corpus)

	print(topic_vec,train_vec)
	exit()

	y_pred = fit(train_vec,df)
	
	constructConfusionMatrix(accuracy(Y_test,y_pred),args.modeltype)




	
	#print(text)


def main():
	dataframe = read_Dataframe("../Processed_DATA/review_prepared_lemma.csv")

	dataframe = dataframe.head(1000)
	pipeline(dataframe)

#main()







