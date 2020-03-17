import pandas as pd 
import numpy as np

df = pd.read_csv("./MSFT_US.csv")
#shift dataframe from date asc to date desc

df["Intraday_change"] = df["Close"] - df["Open"]
df = df.drop("Adjusted_close",axis=1)

df["Price_range"] = df["High"] - df["Low"]

#  .shift(1) takes row with lower index
#  .shift(-1) takes row with higher index

df["Open_change"] = df["Open"] - df["Open"].shift(1)
df["Close_change"] = df["Close"] - df["Close"].shift(1)

for sma_period in [5,10,20,50,100,200]:
    indicator_name = "SMA_%d" % (sma_period)
    df[indicator_name] = df['Close'].rolling(sma_period).mean()

df['BollingerBand_Up_20_2'] = df['Close'].rolling(20).mean() + 2*df['Close'].rolling(20).std()
df['BollingerBand_Down_20_2'] = df['Close'].rolling(20).mean() - 2*df['Close'].rolling(20).std()
df['BollingerBand_Up_20_1'] = df['Close'].rolling(20).mean() + df['Close'].rolling(20).std()
df['BollingerBand_Down_20_1'] = df['Close'].rolling(20).mean() - df['Close'].rolling(20).std()
df['BollingerBand_Up_10_1'] = df['Close'].rolling(10).mean() + df['Close'].rolling(10).std()
df['BollingerBand_Down_10_1'] = df['Close'].rolling(10).mean() - df['Close'].rolling(10).std()
df['BollingerBand_Up_10_2'] = df['Close'].rolling(10).mean() + 2*df['Close'].rolling(10).std()
df['BollingerBand_Down_10_2'] = df['Close'].rolling(10).mean() - 2*df['Close'].rolling(10).std()

for channel_period in [5,10,20,50,100,200]:
    up_name = "Donchian_Channel_Up_%d" % (channel_period)
    down_name = "Donchian_Channel_Down_%d" % (channel_period)
    
    df[up_name] = df['High'].rolling(channel_period).max()
    df[down_name] = df['Low'].rolling(channel_period).min()

forward_lag = 5
df['Target'] = df['Close'].shift(-forward_lag)
df = df.drop('Close',axis=1)
df = df.dropna()
df.to_csv(r"./modified_MSFT.csv", index = False)