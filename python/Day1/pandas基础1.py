import pandas as pd
import numpy as np

#查询pandas版本
print(pd.__version__)
print(pd.show_versions())

#头尾相连两个series
ser1 = pd.Series([1, 2, 3, 4, 5])
ser2 = pd.Series([4, 5, 6, 7, 8])
c = pd.concat([ser1, ser2], axis=0)
print(c)

#找出在序列A中而不在序列B中的元素
print(ser1[~ser1.isin(ser2)])
