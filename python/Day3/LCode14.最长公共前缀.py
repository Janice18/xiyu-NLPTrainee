"""
思路：第一步：找出最短的字符串
     第二步：把第一个字符串默认为最长，依此与长度最短的字符串比较
"""

class Solution(object):
    def longestSubFix(self,strs):
        if not strs:
            return ""
        if len(strs) == 1:
            return strs[0]
        minl = min([len(x) for x in strs])
        end = 0
        while end < minl:
            for i in range(1, len(strs)):
                if strs[i][end] != strs[i - 1][end]:
                    return strs[0][:end]
            end += 1
        return strs[0][:end]


def main():
    a = ['flower', 'flow', 'flight']
    b = ['dog', 'racecar', 'car']
    print(Solution().longestSubFix(a))
    print(Solution().longestSubFix(b))

if __name__ == '__main__':
    main()

