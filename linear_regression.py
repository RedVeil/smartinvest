import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from bokeh.plotting import figure, output_file, show
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

#  .shift(1) takes row with lower index
#  .shift(-1) takes row with higher index

def clean_data(df):
    df["Intraday_change"] = df["Open"]/df["Close"] 
    df["Close"] = df["Adjusted_close"]
    df = df.drop("Adjusted_close",axis=1)
    return df


def create_change_ranges(df):
    df["Maximum_change"] = df["Low"]/df["High"]
    df["Open_change"] = df["Open"].shift(1)/df["Open"]
    df["Close_change"] = df["Close"].shift(1)/df["Close"]
    return df

def create_sma(df):
    for sma_period in [5,10,20,50,100,200]:
        indicator_name = "SMA_%d" % (sma_period)
        df[indicator_name] = df['Close'].rolling(sma_period).mean()
    return df

def create_boilinger(df):
    for period in [10,20]:
        for deviation in [1,2]:
            up_name = f'BollingerBand_Up_{period}_{deviation}'
            down_name = f'BollingerBand_Down_{period}_{deviation}'
            df[up_name] = df['Close'].rolling(period).mean() + deviation*df['Close'].rolling(period).std()
            df[down_name] = df['Close'].rolling(period).mean() - deviation*df['Close'].rolling(period).std()
    return df

def create_donchian_channel(df):
    for channel_period in [5,10,20,50,100,200]:
        up_name = "Donchian_Channel_Up_%d" % (channel_period)
        down_name = "Donchian_Channel_Down_%d" % (channel_period)  
        df[up_name] = df['High'].rolling(channel_period).max()
        df[down_name] = df['Low'].rolling(channel_period).min()
    return df

def normalize(df):
    blub = ["Volume","Intraday_change","Maximum_change","Open_change","Close_change"]
    for column in df.columns.values:
        if column not in blub:
            print(column)
            df[column] = (df[column] - df["Donchian_Channel_Down_200"])/(df["Donchian_Channel_Up_200"]-df["Donchian_Channel_Down_200"])
        else:
            df[column] = (df[column] - df[column].min() ) / (df[column].min()-df[column].max())
    return df


def create_target(df, forward_lag=5):
    df['Target'] = df['Close'].shift(-forward_lag)
    df = df.dropna()
    df2 = df.copy()
    df = df.drop(['Close',"Date"],axis=1) 
    return df, df2

def create_indicator(df):
    df = create_sma(df)
    df = create_boilinger(df)
    df = create_donchian_channel(df)
    return df

if __name__ == "__main__":
    df = pd.read_csv("./MSFT_US.csv")
    #df2 = pd.read_csv("./AAPL_US.csv")
    #df = df.append(df2)
    df = clean_data(df)
    df = create_change_ranges(df)
    df = create_indicator(df)
    df,df2 = create_target(df,7)
    #print(df.head())
    #df = normalize(df)
    x = df.drop("Target",axis=1)
    y = df['Target']
    cut_off = int(len(x)-(len(x)*0.2))
    #x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    x_train = x[:cut_off]
    x_test = x[cut_off:]
    y_train = y[:cut_off]
    y_test = y[cut_off:]
    mlr = LinearRegression()
    mlr.fit(x_train, y_train)
    y_predict = mlr.predict(x_test)
    coefficients = mlr.coef_
    print("Feature Coefficients")
    for i in range(len(coefficients)):
        print(f"{x_train.columns.values[i]} : {coefficients[i]}")
    print("Mean Absolute Error")
    print(mean_absolute_error(y_test,y_predict))
    print("Train score:")
    print(mlr.score(x_train, y_train))
    print("Test score:")
    print(mlr.score(x_test, y_test))
    print("Predicted Values")
    print(y_predict)
    plt.scatter(y_test,y_predict)
    plt.xlabel("Real")
    plt.ylabel("Predicted")
    plt.title("Linear regression")
    plt.show()

    window_size = 30
    window = np.ones(window_size)/float(window_size)
    output_file("stocks.html", title="prediction vs real")

    p = figure(plot_width=1200, plot_height=525, x_axis_type="datetime")
    # add renderers
    p.line(range(len(y_predict)), y_predict, color='red', legend_label='prediction')
    p.line(range(len(y_predict)), df2[cut_off:]["Close"], color='blue', legend_label='real')
    # NEW: customize by setting attributes
    p.title.text = "Prediction vs Real"
    p.legend.location = "top_left"
    p.grid.grid_line_alpha = 0.1
    p.xaxis.axis_label = 'Date'
    p.yaxis.axis_label = 'Price'
    show(p)

    data = {"Date":df2[cut_off:]["Date"], "Prediction_close" :y_predict}
    predict_df = pd.DataFrame(data)
    print(predict_df.head())
    predict_df.to_csv(r"./prediction_MSFT.csv", index = False)

    # show the results
    