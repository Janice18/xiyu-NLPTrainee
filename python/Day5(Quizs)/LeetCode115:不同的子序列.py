"""
方法：动态规划
思路：dp[i][j]代表T前i字符串可以由S前j字符串组成最多个数，
其中T表示决策阶段，S表示问题状态
动态方程为：
当 S[j] == T[i], dp[i][j] = dp[i-1][j-1] + dp[i][j-1];
当S[j] != T[i], dp[i][j] = dp[i][j-1]
其中，对于第一行, T为空,因为空集是所有字符串子集, 所以我们第一行都是1
对于第一列, S长度为0,这样组成T个数为0
"""
class Solution:
    def numDistinct(self, s: str, t: str) -> int:
        if t == "" and s == "":
            return 0
        if t == "":
            return 1
        dp = [[0 for i in range (len(s)+1)] for j in range(len(t)+1) ]  #构建矩阵并初始化
        for j in range(0,len(s)+1):
            dp[0][j] = 1    #决策阶段为0的对应的问题状态输出都为1
        for i in range(1,len(t)+1):   #遍历索引从1开始的所有阶段
            for j in range(1,len(s)+1):    #遍历索引从1开始的问题状态
                dp[i][j] = dp[i][j-1]
                if s[j-1] == t[i-1]:
                    dp[i][j] += dp[i-1][j-1]
        return dp[-1][-1]

if __name__ == '__main__':
    s1, t1 = 'rabbbit','rabbit'
    s2, t2 = 'babgbag','bag'
    print(Solution().numDistinct(s1,t1))
    print(Solution().numDistinct(s2,t2))
