"""
方法：动态规划
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

if __name__ == '__main__':
    print(Solution().minCut('aab'))

#表示2
class Solution(object):
    def minCut(self, s: str) -> int:
        n = len(s)
        dp = [0 for __ in range(n)]
        isPal = [[False for __ in range(n)] for __ in range(n)] # 是否回文存储矩阵
        for i in range(n):
            m = i
            for j in range(i + 1):  # j在左边开始，i右边
                if s[j] == s[i] and (j + 1 > i - 1 or isPal[j + 1][i - 1]):  # 如果i和j都不相等，不需要去判断是否是回文
                    isPal[j][i] = True
                    if j == 0:  # 整个都是回文串
                        m = 0
                    else:
                        m = min(m, dp[j - 1] + 1)  # 要么每个字母都拆，要么之前的字母拆了后+1
            dp[i] = m

        return dp[-1]
 
 if __name__ == '__main__':
    print(Solution().minCut('aab'))
