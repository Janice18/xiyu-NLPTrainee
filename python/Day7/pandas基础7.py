import pandas as pd
import numpy as np

#如何将numpy数组转换为给定形状的dataframe
ser = pd.Series(np.random.randint(1,10,35))
df = pd.DataFrame(ser.values.reshape(7,5))
print(df)


#从dataframe中获取c列最大值所在的行号
df = pd.DataFrame({'c':[1,2,3,4]*25})
print(np.where(df == np.max(df)))
