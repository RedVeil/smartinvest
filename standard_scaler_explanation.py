import numpy as np 
from sklearn import metrics, preprocessing

data = np.arange(20/1)
data_std = np.std(data,ddof=0)
diff = np.diff(data)
diff = np.insert(diff, 0, 0)
diff_std = np.std(diff,ddof=0)
xdata = np.column_stack((data, diff))
print(f"data: \n{xdata}")
scaler = preprocessing.StandardScaler()
scaler.fit(xdata)  #_transform
print(f"transformed_data: \n{scaler.transform(xdata)}")
print(f"means: {scaler.mean_}")
print(f"standard_deviation: {[data_std, diff_std]}")
print(f"Proof Row1: {[(data[0]-scaler.mean_[0])/data_std, (diff[0]-scaler.mean_[1])/diff_std]}")