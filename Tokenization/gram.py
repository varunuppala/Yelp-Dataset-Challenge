from nltk.util import ngrams
from pprint import pprint
import pandas as pd
from collections import Counter
from textblob import TextBlob
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))


def unigrams():

	for i in range(1,5):
		print("------------------1 STAR---------------------")
		df = pd.read_csv("stars_data/"+str(i)+"star.csv")
		print(len(df))
		df = df[df.business_id != '#NAME?']
		df = df[df.review_id != '#NAME?']
		df = df[df.user_id != '#NAME?']

		print("FINAL LENGTH: ", len(df))

		df = df.dropna()
		df = df.reset_index()

		f = open("unigrams.txt", "a")

		for index, row in df.iterrows():
			text = row['text']
			blob = TextBlob(text)
			bigrams = blob.ngrams(n=1)

			for item in unigrams:
			    print(item[0]+" "+item[1])
			    f.write(str(item[0]+" "+item[1]))
	f.close()

def bigrams():

	for i in range(1,6):
		print("------------------%s STAR---------------------"%i)
		
		df = pd.read_csv("../Processed_DATA/stars_data/"+str(i)+"star.csv")
		
		print(len(df))
		df = df[df.business_id != '#NAME?']
		df = df[df.review_id != '#NAME?']
		df = df[df.user_id != '#NAME?']

		print("FINAL LENGTH: ", len(df))

		df = df.dropna()
		df = df.reset_index()

		
		f = open("../Processed_DATA/stars_data/"+str(i)+"star_bigrams.txt", "a")

		df['text'] = df['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
		"""
		for index, row in df.iterrows():

			blob = TextBlob(row['text'])
			bigrams = blob.ngrams(n=2)

			for item in bigrams:
			    print(item[0]+" "+item[1])
			    f.write(str(item[0]+" "+item[1])+"\n")


		f.close()
		"""
		df.to_csv("../Processed_DATA/stars_data_stopwords/"+str(i)+"star.csv")	

#unigrams()
bigrams()