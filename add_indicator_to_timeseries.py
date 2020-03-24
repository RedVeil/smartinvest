import pandas as pd 
import numpy as np

df = pd.read_csv("./MSFT_US.csv")
#shift dataframe from date asc to date desc

df["Intraday_change"] = df["Open"]/df["Close"] 
df["Close"] = df["Adjusted_close"]
df = df.drop("Adjusted_close",axis=1)

df["Maximum_change"] = df["Low"]/df["High"]
df["Open_change"] = df["Open"].shift(1)/df["Open"]
df["Close_change"] = df["Close"].shift(1)/df["Close"]

#  .shift(1) takes row with lower index
#  .shift(-1) takes row with higher index

for sma_period in [5,10,20,50,100,200]:
    indicator_name = "SMA_%d" % (sma_period)
    df[indicator_name] = df['Close'].rolling(sma_period).mean()

for period in [10,20]:
    for deviation in [1,2]:
        up_name = f'BollingerBand_Up_{period}_{deviation}'
        down_name = f'BollingerBand_Down_{period}_{deviation}'
        df[up_name] = df['Close'].rolling(period).mean() + deviation*df['Close'].rolling(period).std()
        df[down_name] = df['Close'].rolling(period).mean() - deviation*df['Close'].rolling(period).std()


for channel_period in [5,10,20,50,100,200]:
    up_name = "Donchian_Channel_Up_%d" % (channel_period)
    down_name = "Donchian_Channel_Down_%d" % (channel_period)  
    df[up_name] = df['High'].rolling(channel_period).max()
    df[down_name] = df['Low'].rolling(channel_period).min()

forward_lag = 5
df['Target'] = df['Close'].shift(-forward_lag)
#df = df.drop('Close',axis=1)
df = df.dropna()
df.to_csv(r"./modified_MSFT.csv", index = False)