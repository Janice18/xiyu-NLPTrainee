"""
方法：正则表达式
思路：利用正则表达式匹配，整个数可分为三部分匹配，拿-13.14e-520举例：
符号（-），正则：[\+\-]?。
e前面的数字（13.14），正则：(\d+\.\d+|\.\d+|\d+\.|\d+)。这里分了4种情况考虑，
且匹配有先后顺序（经调试，0.0，.0，0.，0都是有效的）：
有小数点且小数点前后都有数字；
有小数点且只有小数点前面有数字；
有小数点且只有小数点后面有数字；
没有小数点（整数）。
e及其指数（e-520），正则：(e[\+\-]?\d+)?。e0也有效。
"""
class Solution:
    def isNumber(self, s: str) -> bool:
        import re
        pat = re.compile(r'^[\+\-]?(\d+\.\d+|\.\d+|\d+\.|\d+)(e[\+\-]?\d+)?$')
        return True if len(re.findall(pat, s.strip())) else False

if __name__ == '__main__':
    s = Solution()
    a1 = '-13.14e-520'
    a2 = '--6'
    a3 = '-+3'
    a4 = '95a54e53'
    print(s.isNumber(a1))
    print(s.isNumber(a2))
    print(s.isNumber(a3))
    print(s.isNumber(a4))
