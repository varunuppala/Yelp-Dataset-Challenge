import json
import pandas as pd
data = [] 
with open('yelp_academic_dataset_business.json',encoding='Latin-1') as f: 
	for line in f: 
		j = json.loads(line)
		data.append(j)


count = 0
df1 = pd.DataFrame(columns=['business_id','business_name','star','state','categories','review_count'])
for i in range(len(data)):
	if data[i]['categories'] is not None:
		if ('Restaurants' in data[i]['categories']):
			if (data[i]['state'] == "FL" or data[i]['state'] == "MA"):
				count+=1
				print(count)
				df1.loc[i] = [data[i]['business_id'],data[i]['name'],data[i]['stars'],data[i]['state'],data[i]['categories'],data[i]['review_count']]
df1.to_csv('business_information.csv', index = None)