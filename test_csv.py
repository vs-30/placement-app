import pandas as pd

url = "https://raw.githubusercontent.com/krishnadey30/LeetCode-Questions-CompanyWise/master/amazon_alltime.csv"
df = pd.read_csv(url)
print(df.head())
print(df.columns)
