"""
思路：首先按空格切分数组，然后再翻转数组，最后再用空格连接并转化为字符串形式
"""
class Solution():
    def reverseSentence(self, s: str) -> str:
        return ' '.join(s.split(' ')[::-1])


if __name__ == '__main__':
    a1 = 'I am a student.'
    print(Solution().reverseSentence(a1))


"""
思路二：按空格切分数组，依此入栈，再依此出栈并用空格连接
"""
class Solution():
    def reverseSentence2(self, s: str) -> str:
        stack = []
        ans = ''

        for i in s.split(' '):
            stack.append(i)

        while len(stack) > 0:
            ans += stack.pop() + ' '
        return ans

if __name__ == '__main__':
    a1 = 'I am a student.'
    print(Solution().reverseSentence2(a1))





