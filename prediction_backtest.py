import pandas as pd
import numpy as np 
from bokeh.plotting import figure, output_file, show

msft = pd.read_csv("./modified_MSFT.csv")
predictions = pd.read_csv("./prediction_MSFT.csv")
print(msft.dtypes)