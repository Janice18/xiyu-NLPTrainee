"""
方法：哈希表
"""
class Solution:
    def intToRoman(self, num: int) -> str:
        lookup = {
            1:'I',
            4:'IV',
            5:'V',
            9:'IX',
            10:'X',
            40:'XL',
            50:'L',
            90:'XC',
            100:'C',
            400:'CD',
            500:'D',
            900:'CM',
            1000:'M'
        }
        res = ""
        for key in sorted(lookup.keys())[::-1]:
            a = num // key
            if a == 0:
                continue
            res += (lookup[key] * a)
            num -= a * key
            if num == 0:
                break
        return res

if __name__ == '__main__':
    a1, a2, a3, a4 = 4, 320, 58, 1994
    s = Solution()
    print(s.intToRoman(a1))
    print(s.intToRoman(a2))
    print(s.intToRoman(a3))
    print(s.intToRoman(a4))
