import pandas as pd 

df = pd.read_csv("./AAPL_US.csv")
print(df[-5:]["Date"])