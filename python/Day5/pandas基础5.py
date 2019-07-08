import pandas as pd
import re
#从series中找出包含两个以上元音字母的单词
ser = pd.Series(['Apple', 'Orange', 'Plan', 'Python', 'Money'])

def count(x):
    aims = 'aeiou'
    c= 0
    for i in x:
        if i in aims:
            c += 1
    return c

counts = ser.map(lambda x: count(x))
print(ser[counts>=2])

#如何过滤series中的有效电子邮件
emails = pd.Series(['buying books at amazom.com',
                    'rameses@egypt.com',
                    'matt@t.co',
                    'narendra@modi.com'])
pattern ='[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,4}'
valid = emails.str.findall(pattern, flags=re.IGNORECASE)
b = [x[0] for x in valid if len(x)]
print(b)
