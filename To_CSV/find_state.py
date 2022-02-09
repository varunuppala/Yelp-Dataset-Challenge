import pandas as pd

def extract():
	df = pd.read_csv('business_csv.csv')

	df2 = pd.DataFrame(columns=['business_id','business_name','star','state'])
	for index,row in df.iterrows():
		if row['state'] == 'FL' or row['state'] == 'MA':
			df2 = df2.append(df.iloc[index])

	df2 = df2.reset_index()
	df2.to_csv('FL_MA.csv')

extract()