import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import seaborn as sns

price = np.sin(np.arange(400)/30.0)
price = [i+2 for i in price]
diff = np.diff(price)
diff = np.insert(diff, 0, 0)
print(diff)
'''
positions = [np.random.randint(0,2) for i in price]
data = {"price":price,"position":positions}
df = pd.DataFrame(data=data)
df["strategy_pct"] = df["price"].pct_change(1) * df["position"]
df["strategy"] = (df["strategy_pct"] + 1).cumprod()
df["buy_hold"] = (df["price"].pct_change(1) + 1).cumprod()
print(df)
df[["strategy","buy_hold"]].plot()
plt.savefig('test.png')
plt.show()
'''