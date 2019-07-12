class Solution:
    def zConvert(self, s:str, numRows:int):
        if len(s) <= numRows or numRows == 1:
            return s
        rows = [''] * numRows
        currRow, forward = 0, 1
        for i in s:
            rows[currRow] += i
            if currRow == 0:
                forward = 1
            if currRow == numRows-1:
                forward = -1

            currRow += forward
        return ''.join(rows)


if __name__ == '__main__':
    a = 'LEETCODEISHIRING'
    n1 = 3
    n2 = 4
    print(Solution().zConvert(a, n1))
    print(Solution().zConvert(a, n2))
