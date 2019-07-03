"""
方法：动态规划（动态方程），dim[i][j]表示s1的前(i+1)元素和s2的前(j+1)元素是否交错组成s3前i+j元素
主要考虑四种情况：
1.i=0 & j≠0，则返回dim[i][j] = dim[i][j-1] && s2[j-1] == s3[i+j-1]
2.i≠0 & j=0，则返回dim[i][j] = dim[i-1][j] && s1[i-1] == s3[i+j-1]
3.i=0 & j=0，则返回True
4.i≠0 & j≠0，则返回dim[i][j] = dim[i-1][j] && s1[i-1] == s3[i+j-1] || dim[i][j-1] && s2[j-1] == s3[i+j-1]
"""
class Solution:
    def isInterleave(self, s1, s2, s3):
        """
        :type s1: str
        :type s2: str
        :type s3: str
        :rtype: bool
        """
        s1Len, s2Len, s3Len = len(s1), len(s2), len(s3)
        if s1Len + s2Len != s3Len:
            return False

        # 生成（s1Len+1)*(s2Len+1)的False矩阵
        dim = [[False]*(s1Len + 1) for _ in range(s2Len+1)]

        for i in range(s1Len + 1):
            for j in range(s2Len + 1):
                if i == 0 and j == 0:
                    dim[i][j] = True
                elif i == 0:
                    dim[i][j] = dim[i][j-1] and s2[j-1] == s3[i+j-1]
                elif j == 0:
                    dim[i][j] = dim[i-1][j] and s1[i-1] == s3[i+j-1]
                else:
                    dim[i][j] = dim[i-1][j] and s1[i-1] == s3[i+j-1]\
                        or dim[i][j-1] and s2[j-1] == s3[i+j-1]
        return dim[-1][-1]

def main():
    s1 = 'aabcc'
    s2 = 'dbbca'
    s3 = 'aadbbcbcac'
    s4 = 'aadbbbaccc'
    print(Solution().isInterleave(s1, s2, s3))
    print(Solution().isInterleave(s1, s2, s4))

if __name__ == '__main__':
    main()



