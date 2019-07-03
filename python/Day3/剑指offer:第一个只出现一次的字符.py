"""
思路1：对每个字符出现的字符进行个数统计，然后再对原字符串进行遍历，
找出第一个出现次数为1的字符进行返回即可。
"""
from collections import Counter


class Solution:
    def FirstNotRepeatingChar(self, s):
        if not s: return -1     #不存在，输出-1
        count = Counter(s)
        for i,c in enumerate(s):
            if count[c] == 1:
                return i,c

"""
思路2：利用Python中的字典。字典的键（Key）一定唯一，
每个键对应的值（Value）对应该键Key出现的次数。
"""
class Solution2():
    def FirstNotRepeatingChar(self, s):
        dict = {}
        for ele in s:
            dict[ele] = 1 if ele not in dict else dict[ele] + 1
        for i in range(len(s)):
            if dict[s[i]] == 1:
                return i
        return -1



def main():
    a1 = 'aaaaabbsndddmsq'
    a2 = 'bbmmddjjkalalkmd'
    print(Solution().FirstNotRepeatingChar(a1))
    print(Solution2().FirstNotRepeatingChar(a2))

if __name__ == '__main__':
    main()


