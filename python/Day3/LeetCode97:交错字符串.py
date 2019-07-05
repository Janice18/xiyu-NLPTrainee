"""
方法1：动态规划（动态方程），dim[i][j]表示s1的前(i+1)元素和s2的前(j+1)元素是否交错组成s3前i+j元素
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
 
"""
法二：递归，思路如下：
先检查s3的第一个字符，如果等于s1的第一个字符，且不等于s2的第二个字符，将s1和s3向后移动一位，递归。
当等于s2的第一个字符且不等于与s1的第一个字符的时候，将s2和s3向后移动一位，递归。
如果既等于s1的第一个字符又等于s2的第一个字符，
就有可能有两种情况，s2不动，s1向后移动一位或者是s1不动，s2向后移动一位。
"""
class Solution:
    def isInterleave(self, s1: str, s2: str, s3: str) -> bool:
        if len(s3) != len(s1) + len(s2):
            return False
        if s1 == "" and s2 == "" and s3 == "":
            return True
        if s1 == "":
            return s2 == s3
        if s2 == "":
            return s1 == s3
        if s3[0] == s1[0] and s3[0] != s2[0]:
            return self.isInterleave(s1[1:],s2,s3[1:])
        elif s3[0] == s2[0] and s3[0] != s1[0]:
            return self.isInterleave(s1,s2[1:],s3[1:])
        elif s3[0] == s2[0] and s3[0] == s1[0]:
            return self.isInterleave(s1[1:],s2,s3[1:]) or self.isInterleave(s1,s2[1:],s3[1:])
        else:
            return False
