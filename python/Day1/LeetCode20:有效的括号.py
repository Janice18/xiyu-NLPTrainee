"""
.思路：通过栈实现括号时候匹配，对于左边的括号直接入栈，对于右边的括号就取出栈顶元素进行对比，若不属于同一种类型，
则返回错误，最后判断栈是否为空，若为空则返回正确。
"""


def isValid(s):
    if len(s) % 2 == 1:
        return False
    if len(s) == 0:
        return True
    stack = []
    d = {'{': '}', '[': ']', '(': ')'}
    for i in s:
        if i in d:
            stack.append(i)
        else:
            if not stack or d[stack.pop()] != i:
                return False
    return stack == []


print(isValid('([])'))
print(isValid('{()]'))

#实现方式2
class Solution(object):
    def isValid(self, s):
        d = {')': '(', '}': '{', ']': '['}
        stack = []
        for char in s:
            if char in '({[':
                stack.append(char)
            elif char in ')}]':
                if not stack:
                    return False
                else:
                    if stack.pop() != d[char]:
                        return False
        if stack: #如果stack里面还有内容输出False
            return False
        else:
            return True


s = '(()'
print(Solution().isValid(s))
