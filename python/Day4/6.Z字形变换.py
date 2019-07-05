"""
法一：
思路：从左到右迭代 s，将每个字符添加到合适的行。可以使用当前行和当前方向
这两个变量对合适的行进行跟踪，只有当我们向上移动到最上面的行或向下移动到最下面的行时，
当前方向才会发生改变。
"""
class Solution:
    def convert(self, s: str, numRows: int) -> str:
        if numRows == 1 or len(s) <= numRows:   #考虑baseline
            return s

        rows = [''] * numRows   #初始化一个长度为nurRows，元素为字符串的列表rows。
        currentRow,step = 0, 1
        for i in s:
            rows[currentRow] += i
            if currentRow == 0:   #首行，改变方向
                step = 1
            if currentRow == numRows - 1:
                step = -1

            currentRow += step
        return ''.join(rows)


if __name__ == '__main__':
    a = 'LEETCODEISHIRING'
    n1 = 3
    n2 = 4
    print(Solution().convert(a, n1))
    print(Solution().convert(a, n2))
