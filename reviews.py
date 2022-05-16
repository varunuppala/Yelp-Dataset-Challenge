import json
import pandas as pd


FL_CA = pd.read_csv('business_information.csv')
business_id_list = FL_CA['business_id'].tolist()

data = [] 
with open('yelp_academic_dataset_review.json',encoding='Latin-1') as f: 
	for line in f: 
		j = json.loads(line)
		data.append(j)


df = pd.DataFrame(columns=['business_id','review_id','user_id','stars','text','date'])
count = 0
for i in range(len(data)):
	if (data[i]['business_id'] in business_id_list):
		count +=1
		print(count)
		df.loc[i] = [data[i]['business_id'],data[i]['review_id'],data[i]['user_id'],data[i]['stars'],data[i]['text'],data[i]['date']]

print(df)
df.to_csv('review_information.csv')