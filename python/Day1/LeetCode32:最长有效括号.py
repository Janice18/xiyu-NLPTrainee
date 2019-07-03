"""
方法1：栈
"""
class Solution(object):
    def longestValidParentheses(self, s):
        tl = len(s)
        stack = []
        st = 0
        maxlen = 0
        for i in range(tl):
            # 如果是左括号，直接入stack
            if s[i] == '(':
                stack.append(i)
            # 如果右括号
            else:
                # 如果stack里没有元素匹对，说明有效括号已经结束，更新起始位置
                if len(stack) == 0:
                    st = i + 1
                    continue
                # 有元素匹对
                else:
                    a = stack.pop()
                    # pop出一个左括号匹对
                    # 如果此时没了，不能保证不继续有效括号，所以根据当前的最长距离去更新maxlen
                    if len(stack) == 0:
                        maxlen = max(i - st + 1, maxlen)
                    # 如果此时还有 则计算与栈顶的索引相减来计算长度
                    else:
                        maxlen = max(i - stack[-1], maxlen)

        return maxlen

"""
方法2：动态规划(dp数组)
"""
class Solution:
    def longestValidParentheses(self, s):
        a = len(s)
        if a < 2:
            return 0
        maxlen = 0
        # dp[i]表示刚好在s[i]以前（包括s[i]在内)的最长括号长度
        # 如果s[i] = '(', dp[i] = 0
        dp = [0 for _ in range(a)]
        for i in range(1, a):
            # 当前i的对称点的索引
            pos = i - 1 - dp[i - 1]
            if s[i] == ')' and s[i - 1] == '(' and i - 2 >= 0:
                # 等于前前项的最长长度+2
                dp[i] = dp[i - 2] + 2
            elif s[i] == ')' and pos >= 0 and s[pos] == '(':
                # 等于上个对称点+2
                dp[i] = dp[i - 1] + 2
                if pos - 1 >= 0:
                    # 如果对称点前还有长度，则加上那段长度
                    dp[i] += dp[i - dp[i - 1] - 2]
        return max(dp)

print(Solution().longestValidParentheses('(()))'))
