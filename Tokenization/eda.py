import gensim
import logging
import pandas as pd
import numpy as np
import argparse
from textblob import TextBlob
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from pprint import pprint
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
import main
import json
import warnings
warnings.filterwarnings('ignore')





# get ratings spread
# get average length of reviews
# top 10 word counts

def ratingSpread(data):
	"""
	find rating spread
	"""
	df = pd.read_csv(data)
	value_df = df['stars'].value_counts()
	plt.figure(figsize=(10,5))
	sns.barplot(value_df.index, value_df.values, alpha=0.8)
	plt.title('Rating Distribution')
	plt.ylabel('Number of Ratings', fontsize=12)
	plt.xlabel('Type of Rating', fontsize=12)
	plt.ticklabel_format(style='plain', axis='y')
	plt.savefig('../images/ratings.png')


def getAverageLen(data):
	df = pd.read_csv(data)
	df['len'] = df['text'].apply(lambda x : len(str(x).split()))
	plt.figure(figsize=(10,5))
	sns.set_style("whitegrid")
	sns.boxplot(x = 'stars', y = 'len', data = df)
	plt.title('Rating length Distribution')
	plt.ylabel('Length of rating', fontsize=12)
	plt.xlabel('Type of Rating', fontsize=12)
	plt.ylim(0,1000)
	plt.ticklabel_format(style='plain', axis='y')
	plt.savefig('../images/averagelen.png')

def top10words(data):
	df = pd.read_csv(data)
	
	df = df.astype({"text": str}, errors='raise') 	
	
	c = Counter(" ".join(df["text"]).split()).most_common(10)
	word_frequency = pd.DataFrame(c, columns = ['Word', 'Frequency'])
	
	plt.figure(figsize=(10,5))
	sns.barplot(word_frequency.Word, word_frequency.Frequency, alpha=0.8)
	plt.title('Top Words')
	plt.ylabel('Count of words', fontsize=12)
	plt.xlabel('Words', fontsize=12)
	plt.ticklabel_format(style='plain', axis='y')
	plt.savefig('../images/topwords.png')

	pprint(c)

def bigrams(words, bi_min=15, tri_min=10):
    bigram = gensim.models.Phrases(words, min_count = bi_min)
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    return bigram_mod

def makenewdf(dataframe):
	#text = dataframe['processed_text'].tolist()

	dataframe['processed_text'] = dataframe['text'].apply(lambda x : makelist(x))
	new_df = dataframe[['stars','processed_text']]
	return new_df

def makelist(text):
	return str(text).split(' ')


def load_lda():
	df = pd.read_csv('../Processed_DATA/review_prepared_lemma.csv')
	
	dataframe = makenewdf(df)
	text = dataframe['processed_text'].tolist()

	print("hello")
	
	bigram_mod = bigrams(text)
	bigram = [bigram_mod[review] for review in text]

	print("hello")

	path = '../Models/'
	lda_model =  gensim.models.ldamodel.LdaModel.load(path+'lda_train.model')
	lda_dict = corpora.Dictionary.load(path+'lda_train.model.id2word')
	bow_corpus = [lda_dict.doc2bow(doc) for doc in bigram]

	lda_viz = gensimvis.prepare(lda_model, bow_corpus, lda_dict)
	pyLDAvis.save_html(lda_viz, 'LDA_Visualization_main.html')

def timevslda():
	x = ['*','**','***','****','*****','Full dataset']
	y = [10156.36,4096.57,3812.28,10547.78,26077.13,62430.03]

	y = [i//3600 for i in y]

	fig = plt.figure(figsize = (8, 6))

	# creating the bar plot
	sns.barplot(x, y, alpha = 0.8)

	plt.xlabel("Type of File")
	plt.ylabel("Time Taken in hrs")

	plt.title("Time Comparison")
	plt.savefig("../images/Time Taken for LDA.png")


#load_lda()
#timevslda()

def basic_graph():
	x = ["Affinity","Majority \n Class \n Baseline","Logistic \n Basic","SVM \n basic",
	"Logistic \n with Lemma","SVM \n with Lemma","Logistic \n with \n Bigram \n Lemma",
	"SVM \n with \n Bigram \n Lemma",
	"Logistic \n with \n Bigram(10) \n Lemma","SVM \n with \n Bigram(10) \n Lemma",
	"Logistic \n with \n Bigram(15)","SVM \n with \n Bigram(15)"]
	y = [0.244,0.421,0.676,0.662,0.675,0.661,0.681, 0.665,0.662,0.677,0.681,0.666]
	
	fig = plt.figure(figsize = (18, 9))

	# creating the bar plot
	sns.barplot(x, y, alpha = 0.8,linewidth = 3,
           edgecolor = "black")

	plt.xlabel("Type of Model",fontsize=15)
	plt.ylabel("Accuracy",fontsize=15)
	plt.ylim(0,1)
	plt.title("Accuracy Comparison")
	plt.savefig("../images/Accuracy Comparison.png")

def constructConfusionMatrix(dictonary,modeltype):

	row1 = list(dictonary.keys())
	
	columns = list(dictonary[row1[0]].keys())
	columns.remove('support')
	
	pprint(dictonary)

	final_list = []
	for key,values in dictonary.items():
		inside_list = []
		if key != "accuracy":
			for k,v in values.items():
				if k != "support":
					inside_list.append(v)
			final_list.append(inside_list)


	row1.remove('accuracy')
	df = pd.DataFrame(final_list, columns = columns,index = row1)
	#df.insert(loc=0, column='Type', value= row1)
	plt.figure(figsize=(10,5))
	sns.heatmap(df)
	plt.title(modeltype)
	plt.savefig("../images/heatmaps/%s.png"%modeltype)

	plt.figure(figsize=(5,3))

	with open('../images/values_text/%s'%modeltype, 'w') as fp:
		json.dump(dictonary, fp,indent = 4)


#basic_graph()


d = {'1.0': {'f1-score': 0.1284153005464481,
         'precision': 0.12415614910478427,
         'recall': 0.13297705124174788,
         'support': 3181},
 '2.0': {'f1-score': 0.0043047783039173474,
         'precision': 0.16666666666666666,
         'recall': 0.002180549498473615,
         'support': 2293},
 '3.0': {'f1-score': 0.010400734169470786,
         'precision': 0.14049586776859505,
         'recall': 0.0054002541296060995,
         'support': 3148},
 '4.0': {'f1-score': 0.1574871850801614,
         'precision': 0.2628321805606116,
         'recall': 0.11242603550295859,
         'support': 6422},
 '5.0': {'f1-score': 0.5193536002233778,
         'precision': 0.3979673709548007,
         'recall': 0.7472880674969867,
         'support': 9956},
 'accuracy': 0.34428,
 'macro avg': {'f1-score': 0.1639923196646751,
               'precision': 0.21842364701109168,
               'recall': 0.2000543915739546,
               'support': 25000},
 'weighted avg': {'f1-score': 0.26532674341053497,
                  'precision': 0.2747783910996307,
                  'recall': 0.34428,
                  'support': 25000}}

#constructConfusionMatrix(d,"LDA on 100,000 rows")
basic_graph()











