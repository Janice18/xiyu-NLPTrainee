import pandas as pd
ser  = pd.Series(['01 Jan 2010 11:11:11',
                '02-02-2011 7:13:26',
                 '20120303 5:12:34',
                 '2013/04/04 8:32:10',
                 '2014-05-05 9:11:30'])
                 
#series如何将一日期字符串转化为时间
print(pd.to_datetime(ser))

#series如何从时间序列中提取年/月/天/小时/分钟/秒
date = pd.to_datetime(ser)
print('year: ', date.dt.year,sep='\n')
print('month: ', date.dt.month, sep='\n')
print('day: ', date.dt.day, sep='\n')
print('hour: ', date.dt.hour, sep='\n')
print('minute: ', date.dt.minute, sep='\n')
print('second: ', date.dt.second, sep='\n')
