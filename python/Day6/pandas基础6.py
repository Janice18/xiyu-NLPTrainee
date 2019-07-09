import pandas as pd
import numpy as np

#seriesA以seriesB为分组依据，然后计算分组后的平均值
fruit = pd.Series(np.random.choice(['apple', 'banana', 'carrot'], 10))#从一维数组中随机采样10个
#np.linspace(start,end,num)，不标明num,默认50个
weights = pd.Series(np.linspace(1, 10, 10))    #生成1-10，10个数
print(weights.groupby(fruit).mean())


#如何创建一个以‘2000-01-02’开始,包含10个周六的TimeSeries
#np.randon.randint(low,high,size)
print(pd.Series(np.random.randint(1,10,10),
          pd.date_range('2000-01-02',
                        periods=10,
                        freq='W-SAT')))
