import pandas as pd
import numpy as np

ser1 = pd.Series([1, 2, 3, 4, 5])
ser2 = pd.Series([4, 5, 6, 7, 8])

#两个series的并集
print(np.union1d(ser1,ser2))

#两个series的非共有元素
u = pd.Series(np.union1d(ser1,ser2))
v = pd.Series(np.intersect1d(ser1, ser2))
print(u[~u.isin(v)])      #此处方括号里面的是布尔型，~表示否定

#获得series的最小值，第25百分位数，中位数，第75位和最大值
ser = pd.Series([11,13,25,33,24,36,22,28])
b = np.percentile(ser, q=[0,25,50,75,100])
print(b)
