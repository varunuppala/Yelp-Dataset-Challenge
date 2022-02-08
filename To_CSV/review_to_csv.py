import json
import pandas as pd
data = [] 
with open('yelp_academic_dataset_review.json',encoding='Latin-1') as f: 
	for line in f: 
		j = json.loads(line)
		data.append(j)

df = pd.DataFrame(columns=['review_id','user_id','business_id','star','text'])
for i in range(len(data)):
	df.loc[i] = [data[i]['review_id'],data[i]['user_id'],data[i]['business_id'],data[i]['stars'],data[i]['text']]

print(df)
df.to_csv('review_csv.csv')
