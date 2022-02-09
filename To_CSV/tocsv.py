import json
import pandas as pd


FL_MA = pd.read_csv('FL_MA.csv')
business_id_list = FL_MA['business_id'].tolist()

data = [] 
with open('yelp_academic_dataset_review.json',encoding='Latin-1') as f: 
	for line in f: 
		j = json.loads(line)
		data.append(j)


review_ids = []
user_ids = []
texts = [] 
for i in range(100):
	if data[i]['business_id'] in business_id_list:
		review_ids.append(data[i]['review_id'])
		user_ids.append(data[i]['user_id'])
		texts.append(data[i]['text'])

df = pd.DataFrame()
df['review_id'] = review_ids
df['user_ids'] = user_ids
df['reviews'] = texts
print(df)
df.to_csv('all_information.csv')