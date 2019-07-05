"""
法一：递归，思路如下：
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

def main():
    s1 = 'aabcc'
    s2 = 'dbbca'
    s3 = 'aadbbcbcac'
    s4 = 'aadbbbaccc'
    print(Solution().isInterleave(s1, s2, s3))
    print(Solution().isInterleave(s1, s2, s4))

if __name__ == '__main__':
    main()
