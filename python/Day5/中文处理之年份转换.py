import re

m0 = "在一九四九年新中国成立"
m1 = "比一九九零年低百分之五点二"
m2 = '人一九九六年击败俄军,取得实质独立'

numHash = {}
numHash['零'] = '0'
numHash['一'] = '1'
numHash['二'] = '2'
numHash['三'] = '3'
numHash['四'] = '4'
numHash['五'] = '5'
numHash['六'] = '6'
numHash['七'] = '7'
numHash['八'] = '8'
numHash['九'] = '9'


def change2num(words):
    newword = ''
    for key in words:
        if key in numHash:
            newword += numHash[key]
        else:
            newword += key
    return newword

def Chi2Num(m):
    a = re.findall(u"[\u96f6|\u4e00|\u4e8c|\u4e09|\u56db|\u4e94|\u516d|\u4e03|\u516b|\u4e5d]+\u5e74", m)
    if a:
        print("------")
        print(m)
        for words in a:
            newwords = change2num(words)
            print(words)
            print(newwords)
            m = m.replace(words, newwords)
    return m

if __name__ == '__main__':
    Chi2Num(m0)
    Chi2Num(m1)
    Chi2Num(m2)


