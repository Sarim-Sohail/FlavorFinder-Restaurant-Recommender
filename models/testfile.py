import pandas as pd 


df=pd.read_csv(r"models/potato.csv")

print(df.head())

df["Cuisines"]=df["Cuisines"].str.strip()

df.to_csv("models/updated_zomato.csv",index=False)