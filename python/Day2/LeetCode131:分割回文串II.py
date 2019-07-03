"""
思路：创建一组arr来存储每个位置的初始最小分割次数，其中索引为0的元素指为-1，
在每个存储位置判断是否可以分割，再不断去更新原来arr的值，最后返回arr[-1]即可
"""
class Solution:
    def minCut(self, s: str) -> int:
        """
     :type s: str
     :rtype: int
     """

        arr = [len(s) for i in range(len(s)+1)]
        arr[0] = -1
        for i in range(len(s)):
            for j in range(i + 1):
                if s[j:i + 1] == s[j:i + 1][::-1]:
                    arr[i + 1] = min(arr[i + 1], arr[j] + 1)
        return arr[-1]


print(Solution().minCut('aacbb'))
