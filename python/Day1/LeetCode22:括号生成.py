
"""
思路：如果当前总数目已经达到要求，则直接添加到结果并跳出函数；
      如果当前左括号个数合法（小于n），可以再添加一个左括号；
      如果当前右括号个数合法（小于左括号个数），可以再添加一个右括号。
"""
class Solution(object):
    def generate(self, n):
        ans = []
        def backtrack(S = '', left = 0, right = 0):
            if len(S) == 2 * n:
                ans.append(S)
                return
            if left < n:
                backtrack(S+'(', left+1, right)
            if right < left:
                backtrack(S+')', left, right+1)

        backtrack()
        return ans


a = Solution()
print(a.generate(3))
