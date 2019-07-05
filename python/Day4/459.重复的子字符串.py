"""
法1思路：重复字符串的长度不会超过原字符串的一半，
从第二个字符开始逐个判断是否与第一个字符相等，若相等则截取其前面的所有字符为一个子串，
判断字符串长度能否被子串整除，如可以再将子串复制到原字符串长度，比较两个字符串是否相等。
"""
class Solution:
    def repeatedSubstringPattern(self, s: str) -> bool:
        """
        :type s: str
        :rtype: bool
        """
        if len(s)<2:
            return False
        mid = len(s) // 2
        for i in range(1, mid + 1):   #重复字符串不会超过原字符串的一半
            if s[i] == s[0] and s[:i]*(len(s)//i) == s:
                return True
        return False

    """
    法2思路：假设母串S是由子串s重复N次而成， 则 S+S则有子串s重复2N次，
     现在S=ns， S+S=2ns 因此S在S+S[1:-1]（即去掉首尾字符串）中必出现一次以上
    """
    def repeatedSubstringPattern(self, s):
        """
        :type s: str
        :rtype: bool
        """
        return (s + s)[1:len(s) * 2 - 1].count(s) != 0

    """
    法3和法2思路一样，表示不同
    """
    def repeatedSubstringPattern(self, s: str) -> bool:
            return s in (s + s)[1: len(s) * 2 - 1]


def main():
    q1 = 'aba'
    q2 = 'abcabcabcabc'
    q3 = 'a'
    print(Solution().repeatedSubstringPattern(q1))
    print(Solution().repeatedSubstringPattern(q2))
    print(Solution().repeatedSubstringPattern(q3))

if __name__ == '__main__':
    main()