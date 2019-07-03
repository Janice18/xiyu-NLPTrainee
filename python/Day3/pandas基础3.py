#series中计数排名前2的元素
ser = pd.Series([2,1,3,4,3,2,3,1,2,2])
cnt = ser.value_counts()
cnt2 = cnt.index[:2]
print(cnt2)

#如何将数字系列分成10个相同大小的组
ser3 = pd.Series([1,2,3,4,5,6,7,8,9,10,11,12,13,12,13,14,15,16,17,20])
groups = pd.qcut(ser3, q=[0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1])
labels = ['1st','2nd','3nd','4th','5th','6th','7th','8th','9th''10th']
print(groups)

#如何将系列中每个元素的第一个字符转换为大写
ser = pd.Series(['how','to','pick','apples?'])
print(ser.map(lambda x:x.title()))
