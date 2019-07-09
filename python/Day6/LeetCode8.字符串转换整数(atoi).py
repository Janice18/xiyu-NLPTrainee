"""
法一：适用正则表达式
用到的点：
^：匹配字符串开头
[\+\-]：代表一个+字符或-字符
?：前面一个字符可有可无
\d：一个数字
+：前面一个字符的一个或多个
\D：一个非数字字符
*：前面一个字符的0个或多个
其中max(min(数字, 2**31 - 1), -2**31) 用来防止结果越界
"""
import re

class Solution:
    def myAtoi(self, s: str) -> int:
        return max(min(int(*re.findall('^[+\-]?\d+', s.lstrip())),2**31-1 ), -2**31)

"""
正则化方法表示二
如果正则表达式中的三组括号把匹配结果分成三组
group() 同group（0）就是匹配正则表达式整体结果
group(1) 列出第一个括号匹配部分，group(2) 列出第二个括号匹配部分，
group(3) 列出第三个括号匹配部分。groups() 返回返回一个包含所有小组字符串的元组，从 1 到 所含的小组号。

"""

class Solution:
    def myAtoi(self, str):
        """
        :type str: str
        :rtype: int
        """

        pattern = r"[\s]*[+-]?[\d]+"      #\s代表任意空白符
        match = re.match(pattern, str)
        if match:
            res = int(match.group(0))    #match.group(0)--匹配正则表达式整体结果
            if res > 2 ** 31 - 1:
                res = 2 ** 31 - 1
            if res < - 2 ** 31:
                res = - 2 ** 31
        else:
            res = 0
        return res


if __name__ == '__main__':
    print(Solution().myAtoi('  -432+12'))
    print(Solution().myAtoi('*&-23'))
    print(Solution().myAtoi('-91283472332'))


