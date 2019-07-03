#法1：非递归
class Solution():
    def letterCombinatins(self,digits):
        if not digits:
            return []

        phoneNo = {
            '2':'abc',
            '3':'def',
            '4':'ghi',
            '5':'jkl',
            '6':'mno',
            '7':'pqrs',
            '8':'tuv',
            '9':'wxyz',
        }

        arr = [i for i in phoneNo[digits[0]]]

        for i in digits[1:]:
            arr = [u+v for u in arr for v in phoneNo[i]]
            print(arr)


if __name__ == '__main__':
    Solution().letterCombinatins('23')

#法2：递归
class Solution(object):
    def letterCombinations(self, digits):
        """
        :type digits: str
        :rtype: List[str]
        """
        # 创建字母对应的字符列表的字典
        dic = {2: ['a', 'b', 'c'],
               3: ['d', 'e', 'f'],
               4: ['g', 'h', 'i'],
               5: ['j', 'k', 'l'],
               6: ['m', 'n', 'o'],
               7: ['p', 'q', 'r', 's'],
               8: ['t', 'u', 'v'],
               9: ['w', 'x', 'y', 'z'],
               }
        # 存储结果的数组
        ret_str = []
        if len(digits) == 0: return []
        # 递归出口，当递归到最后一个数的时候result拿到结果进行for循环遍历
        if len(digits) == 1:
            return dic[int(digits[0])]
        # 递归调用
        result = self.letterCombinations(digits[1:])
        # result是一个数组列表，遍历后字符串操作，加入列表
        for r in result:
            for j in dic[int(digits[0])]:
                ret_str.append(j + r)
        return ret_str



