import pandas as pd
import string
import math
import contractions
from nltk.corpus import stopwords
import argparse
import nltk
from nltk.stem import WordNetLemmatizer 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from pprint import pprint
import eda
import seaborn as sns
import matplotlib.pyplot as plt
import json
from afinn import Afinn
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import LDA
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix,ConfusionMatrixDisplay
import warnings
warnings.filterwarnings('ignore')


def basic_graph():
	x = ["Majority \n Class \n Baseline","Logistic \n Basic","SVM \n basic",
	"Logistic \n with Lemma","SVM \n with Lemma","Logistic \n with Bigram Lemma",
	"SVM \n with Bigram Lemma"]
	y = [0.421,0.676,0.662,0.675,0.661,0.681, 0.665]
	
	fig = plt.figure(figsize = (8, 6))

	# creating the bar plot
	sns.barplot(y, x, alpha = 0.8)

	plt.ylabel("Type of Model")
	plt.xlabel("Accuracy")
	plt.xlim(0,1)
	plt.title("Accuracy Comparison")
	plt.savefig("../images/Accuracy Comparison.png")



# Use below line if required
#nltk.download('omw-1.4')

#### Intializing for below cases #######

tokenizer = nltk.RegexpTokenizer(r"\w+")

lemmatizer = WordNetLemmatizer()

stop_words = set(stopwords.words('english'))

tf_idf = TfidfVectorizer()



def regular_Tokenize(review):
	"""
	To remove punctuations
	"""
	return ' '.join(tokenizer.tokenize(review))

def lowercase(review):
	"""
	Lower case the sentence
	"""
	return review.lower()

def removestopwords(review):
	"""
	Remove the stopwords
	"""
	return ' '.join([word for word in review.split() if word not in (stop_words)])

def resolve_Contractions(review):
	"""
	Resolving all the contractions
	i've = i have
	i'm = i am
	"""
	return contractions.fix(review)

def lemmatize(review):
	"""

	"""
	word_list = nltk.word_tokenize(review)
	return ' '.join([lemmatizer.lemmatize(w) for w in word_list])

def opencsv(file):
	return pd.read_csv(file)

def prepare_corpus(args):

	print(">Preparing for reading the file")

	reviews = opencsv(args.data)
	
	reviews = reviews.dropna()

	reviews['text'] = reviews['text'].apply(lambda x : resolve_Contractions(x))

	print("=>contractions resolved")
	
	reviews['text'] = reviews['text'].apply(lambda x : lowercase(x))

	print("==>lowercasing complete ")	

	reviews['text'] = reviews['text'].apply(lambda x : removestopwords(x))

	print("===>stopwords removed ")

	reviews['text'] = reviews['text'].apply(lambda x : regular_Tokenize(x))

	print("====>punctuation removal complete")



	if args.use_lemma:

		reviews['text'] = reviews['text'].apply(lambda x : lemmatize(x))

		print("=====>lemmatization complete")

	
	if args.use_lemma:
		reviews.to_csv(args.data[:-4]+"_prepared"+"_lemma_5"+".csv")
	else:
		reviews.to_csv(args.data[:-4]+"_prepared_2"+".csv")
	print("!====saved to disk====!")

def tfidf(X_train,X_test):
	tf_x_train = tf_idf.fit_transform(X_train)
	tf_x_test = tf_idf.transform(X_test)

	return tf_x_train,tf_x_test


def trainAndTest(df):
	
	X_train,X_test,Y_train,Y_test = train_test_split(df['processed_text'], df['stars'],
														 test_size=0.25, random_state=30)
	return X_train,X_test,Y_train,Y_test

def SVM_ML(tf_train,Y_train,tf_test):
	clf = LinearSVC(random_state=0)
	
	clf.fit(tf_train,Y_train)
	
	y_test_pred=clf.predict(tf_test)
	
	return clf,y_test_pred

def LR_ML(tf_train,Y_train,tf_test):
	clf = LogisticRegression(max_iter=1000,solver ='saga')
	
	clf.fit(tf_train,Y_train)
	
	y_test_pred=clf.predict(tf_test)
	
	return clf,y_test_pred

def Baseline(tf_train,Y_train,tf_test):
	print(max(Y_train))

	return len(tf_test)*[max(Y_train)]

def accuracy(Y_test,y_test_pred):
	report=classification_report(Y_test, y_test_pred,output_dict=True)
	return report

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



def afinnSent(args,afinn):
	df = pd.read_csv(args.cleandata)

	df = df.astype({"text": str}, errors='raise') 

	df['afinn_score'] = df['text'].apply(afinn.score)

	df.to_csv('../Processed_DATA/review_prepared_affin.csv')


def vaderSent(args,analyzer):
	df = pd.read_csv(args.cleandata)

	df = df.astype({"text": str}, errors='raise') 	

	sentiment = df['text'].apply(analyzer.polarity_scores)

	sentiment_df = pd.DataFrame(sentiment.tolist())

	df_sentiment = pd.concat([df,sentiment_df], axis = 1)

	df_sentiment.to_csv('../Processed_DATA/review_prepared_vader.csv')


def bigrams(text):
	i = 0
	newtext = []
	for j in range(1,len(text)):
		newtext.append(text[i]+text[j])
		i+=1
	return newtext



def makeBigrams(cleandata):
	df = pd.read_csv(cleandata)

	#df = df.head(100)

	print("...")

	df = df.astype({"text": str}, errors='raise') 
	dataframe = LDA.makenewdf(df)

	text = dataframe['processed_text'].tolist()

	print("...")

	bigram_mod = LDA.bigrams(text)
	bigram = [bigram_mod[review] for review in text]

	for i in range(len(bigram)-1):
		bigram[i] = ' '.join(bigram[i])

	dataframe['processed_text'] = bigram

	print("...")

	dataframe.to_csv("review_prepared_bigram_15.csv")


def predict(args):
	# Open dataframe
	df = opencsv(args.cleandata)
	#df = df.head(100)

	df = df.dropna()
	
	print("opened dataset")

	# Get test and train split
	X_train,X_test,Y_train, Y_test = trainAndTest(df)

	print("Got test and train split")

	# tf-idf
	tf_train,tf_test = tfidf(X_train,X_test)

	print("tfidf done")

	print("n_samples: %d, n_features: %d" % tf_train.shape)


	# LR pred
	#clf,y_pred_LR = LR_ML(tf_train,Y_train,tf_test)
	
	#y_pred_baseline = Baseline(X_train,Y_train,X_test) 
	#constructConfusionMatrix(accuracy(Y_test,y_pred_LR),args.modeltype+"_LR")

	
	#error_bar(Y_test,y_pred_LR,"LR")
	#nnmatrix(clf,Y_test,y_pred_LR,"LR")


	# SVM pred
	clf,y_pred_SVM = SVM_ML(tf_train,Y_train,tf_test) 

	constructConfusionMatrix(accuracy(Y_test,y_pred_SVM),args.modeltype+"_SVM")
	
	error_bar(Y_test,y_pred_SVM,"SVM")
	nnmatrix(clf,Y_test,y_pred_SVM,"SVM")

def error_bar(y_test,y_pred,name):
	d = {}
	for x,y in zip(y_test,y_pred):
		a = x - y
		if a not in d:
			d[a] = 0
		d[a]+=1

	x = list(d.keys())
	y = list(d.values())

	print(x,y)
	sns.barplot(x, y, alpha = 0.8)

	plt.ylabel("Frequency")
	plt.xlabel("Error Value")
	#plt.xlim(0,1)
	plt.title("Error Analysis %s"%name)
	plt.savefig("../images/Error_Analysis %s.png"%name)



def nnmatrix(clf,y_test,y_pred,name):
	#Generate the confusion matrix
	cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
	disp = ConfusionMatrixDisplay(confusion_matrix=cm,
	                          display_labels=clf.classes_)

	print(cm)
	disp.plot()
	plt.savefig('../images/error_%s.png'%name)

def main(args):

	if args.prepare_corpus:
		prepare_corpus(args)

	if args.use_predict:
		predict(args)

	if args.use_eda:
		print("Yes")
		if args.eda_ratings:
			#pass cleaned data
			eda.ratingSpread(args.cleandata)
		if args.eda_length:
			# pass cleaned data
			eda.getAverageLen(args.cleandata)
		if args.eda_topwords:
			# pass cleaned data
			eda.top10words(args.cleandata)


	if args.use_affin:
		print("yes")
		afinn = Afinn(language='en')

		analyzer = SentimentIntensityAnalyzer()
		
		vaderSent(args,analyzer)
		#afinnSent(args,afinn)
	if args.make_bigrams:
		print("...")
		makeBigrams(args.cleandata)

if __name__ == '__main__':
	
	args = argparse.ArgumentParser(description='Program description.')
	args.add_argument('-preparecorpus','--prepare_corpus', type = bool, help='prepare_corpus', required=False)
	args.add_argument('-uncleandata','--data', type=str, help='CSV file', required=False)
	args.add_argument('-modeltype','--modeltype', type=str, help='CSV file', required=False)
	args.add_argument('-cleandata','--cleandata', type=str, help='CSV file', required=False)
	args.add_argument('-usepredict','--use_predict', type = bool, help='predict',required=False)
	args.add_argument('-uselemma','--use_lemma', type = bool, help='Use lemmatizer',required=False)
	args.add_argument('-useeda','--use_eda', type = bool, help='Use EDA',required=False)
	args.add_argument('-findratings','--eda_ratings', type = bool, help='spread of ratings',required=False)
	args.add_argument('-findavglen','--eda_length', type = bool, help='average length of reviews',required=False)
	args.add_argument('-findtopwords','--eda_topwords', type = bool, help='top 10 words',required=False)
	args.add_argument('-useaffin','--use_affin', type = bool, help='use polarity scores',required=False)
	args.add_argument('-makebigrams','--make_bigrams', type = bool, help='make bigrams',required=False)
	args.set_defaults(use_lemma=False)
	args = args.parse_args()
	
	main(args)
