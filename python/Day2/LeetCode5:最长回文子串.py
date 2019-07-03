"""
Solution1:动态规划
"""
class Solution:
    def longestPalindrome(self, s):
        size = len(s)
        if size <= 1:
            return s
        # 二维 dp 问题
        # 状态：dp[l,r]: s[l:r] 包括 l，r ，表示的字符串是不是回文串
        # 设置为 None 是为了方便调试，看清楚代码执行流程
        dp = [[False for _ in range(size)] for _ in range(size)]

        longest_l = 1
        res = s[0]

        # 因为只有 1 个字符的情况在最开始做了判断
        # 左边界一定要比右边界小，因此右边界从 1 开始
        for r in range(1, size):
            for l in range(r):
                # 状态转移方程：如果头尾字符相等并且中间也是回文
                # 在头尾字符相等的前提下，如果收缩以后不构成区间（最多只有 1 个元素），直接返回 True 即可
                # 否则要继续看收缩以后的区间的回文性
                # 重点理解 or 的短路性质在这里的作用
                if s[l] == s[r] and (r - l <= 2 or dp[l + 1][r - 1]):
                    dp[l][r] = True
                    cur_len = r - l + 1
                    if cur_len > longest_l:
                        longest_l = cur_len
                        res = s[l:r+1]    #l到r，不包括r+1
        return res


a = Solution()
print(a.longestPalindrome('abcbabde'))
print(a.longestPalindrome('abbaabde'))


"""
Solution2:中心扩散法
1、如果传入重合的索引编码，进行中心扩散，此时得到的最长回文子串的长度是奇数；
2、如果传入相邻的索引编码，进行中心扩散，此时得到的最长回文子串的长度是偶数。
"""
class Solution:
    def longestPalindrome(self, s):
        size = len(s)
        if size == 0:
            return ''

        # 至少是 1
        longest_palindrome = 1
        longest_palindrome_str = s[0]

        for i in range(size):
            paldmOdd, odd_len = self.paldmCenter(s, size, i, i)
            paldmEven, even_len = self.paldmCenter(s, size, i, i + 1)

            # 当前找到的最长回文子串
            cur_max_sub = paldmOdd if odd_len >= even_len else paldmEven
            if len(cur_max_sub) > longest_palindrome:
                longest_palindrome = len(cur_max_sub)
                longest_palindrome_str = cur_max_sub

        return longest_palindrome_str

    def paldmCenter(self, s, size, left, right):
        """
        left = right 的时候，此时回文中心是一条线，回文串的长度是奇数
        right = left + 1 的时候，此时回文中心是任意一个字符，回文串的长度是偶数
        """
        lf = left
        r = right

        while lf >= 0 and r < size and s[lf] == s[r]:
            lf -= 1
            r += 1
        return s[lf + 1:r], r - lf + 1


a = Solution()

print(a.longestPalindrome('abcbabde'))
print(a.longestPalindrome('abbaabde'))
