import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from bokeh.plotting import figure, output_file, show
from bokeh.layouts import column, row
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.utils import shuffle
from data_preperation import prepare_data

if __name__ == "__main__":
    forward_lag = 7
    x_train, x_test, y_train, y_test, df2, cut_off = prepare_data("MSFT",forward_lag=forward_lag,set_shuffle=True)  
    
    model = LinearRegression()
    model.fit(x_train, y_train)
    
    y_predict = model.predict(x_test)
    coefficients = model.coef_
    print("Feature Coefficients")
    for i in range(len(coefficients)):
        print(f"{x_train.columns.values[i]} : {coefficients[i]}")
    print("Mean Absolute Error")
    print(mean_absolute_error(y_test,y_predict))
    print("Train score:")
    print(model.score(x_train, y_train))
    print("Test score:")
    print(model.score(x_test, y_test))
    print("Predicted Values")
    print(y_predict)

    '''
    sma_period = 7
    y_predict_df = pd.DataFrame(data={"index":range(len(y_predict)),"prediction":y_predict})
    print(y_predict_df.head())
    y_predict_df["sma"] = y_predict_df["prediction"].rolling(sma_period).mean()
    y_predict_df = y_predict_df.dropna()
    predict_sma = y_predict_df["sma"]
    print(predict_sma)
    '''
    '''
    plt.scatter(y_test,y_predict)
    plt.xlabel("Real")
    plt.ylabel("Predicted")
    plt.title("Linear regression")
    plt.show()

    '''
    window_size = 30
    window = np.ones(window_size)/float(window_size)
    output_file("linear_regression_shuffle.html", title="prediction vs real")

    p = figure(plot_width=1080, plot_height=472, x_axis_type="datetime")
    # add renderers
    p.line(df2[cut_off+forward_lag:]["Date"], y_predict[:-forward_lag], color="red", legend_label='Prediction')
    p.line(df2[cut_off+forward_lag:]["Date"], df2[cut_off+forward_lag:]["Close"], color="blue", legend_label='Real')
    p.line(df2[cut_off+forward_lag:]["Date"], df2[cut_off+forward_lag:]["SMA_10"], color="darkgreen" ,legend_label='Sma_10')
    # NEW: customize by setting attributes
    p.title.text = "Prediction vs Real"
    p.legend.location = "top_left"
    p.grid.grid_line_alpha = 0.1
    p.xaxis.axis_label = 'Date'
    p.yaxis.axis_label = 'Price'

    p2 = figure(plot_width=1080, plot_height=472, x_axis_type="datetime")
    p2.line(df2[cut_off+forward_lag:]["Date"], y_predict[:-forward_lag]/df2[cut_off+forward_lag:]["Close"], color="red", legend_label='Correlation')
    p2.title.text = "Correlation Prediction vs Real"
    p2.legend.location = "top_left"
    p2.grid.grid_line_alpha = 0.1
    p2.xaxis.axis_label = 'Date'
    p2.yaxis.axis_label = 'Correlation'
    show(row(p,p2))
    
    '''
    data = {"Date":df2[cut_off+5:]["Date"], "Prediction_close" :y_predict[:-5]}
    predict_df = pd.DataFrame(data)
    print(predict_df.head())
    predict_df.to_csv(r"./prediction_MSFT_normalized.csv", index = False)
    '''