class Solution:
    def strToInt(self, s):
        import re
        pattern = r'[\s]*[+-]?[\d]+'
        match = re.match(pattern,s)
        if match:
            res = int(match.group())
            if res > 2**31-1:
                res = 2**31-1
            elif res < -2**31:
                res = -2**31
        else:
            res = 0

        return res


if __name__ == '__main__':
    print(Solution().strToInt('  -432+12'))
    print(Solution().strToInt('*&-23'))
    print(Solution().strToInt('-91283472332'))
