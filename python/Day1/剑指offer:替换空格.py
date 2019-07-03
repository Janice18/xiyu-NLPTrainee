"""
法1：特别做法，只是python适用
因为在Python中字符串不可变，所以定义了一个新的res来存储替换后的字符串，依次遍历字符串，
"""
class Solution:
    # s 源字符串
    def replaceSpace(self, s):
        res = ''
        for i in s:
            if i == ' ':
                res += '%20'
            else:
                res += i
        return res


"""
法2：通用做法
1）计算字符串中有多少个空格，每一个空格需要多增加两个空间。比如字符串’we are happy.’在python中长度为13，替换后的长度为17。（在C语言中包含字符串结尾符合’\0’，所以在C语言中为字符串长度为14）
2)定义两个指针p1,p2，p1指向原始字符串的末尾，p2指向替换之后的字符串的末尾.接下来向前移动p1，逐个把它指向的字符复制到p2的位置，同时p2前移。当p1遇到空格时，p2前移三格插入’%20’。
3）当p1、p2指向同一位置时，说明所有空格已经被替换。
"""
class Solution(object):
    def replaceSpace(self, s):
        # write code here
        # second solution
        p1 = len(s) - 1
        res = list(s)
        n = s.count(' ')
        res += [0] * n * 2
        p2 = len(res) - 1
        while p1 != p2:
            if res[p1] == ' ':
                res[p2] = '0'
                res[p2 - 1] = '2'
                res[p2 - 2] = '%'
                p2 -= 3
            else:
                res[p2] = res[p1]
                p2 -= 1
            p1 -= 1
        return ''.join(res)


solution = Solution()
Str = "We Are Happy"
mystr = solution.replaceSpace(Str)
print(mystr)
