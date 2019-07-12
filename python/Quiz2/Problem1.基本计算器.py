class Solution:
    def calculate(self, s:str) ->int:
        stack = []
        pre_op = '+'
        num = 0
        for i, each in enumerate(s):
            if each.isdigit():
                num = 10*num + int(each)
            if i == len(s)-1 or each in '+-*/':
                if pre_op == "+":
                    stack.append(num)
                elif pre_op == "-":
                    stack.append(-num)
                elif pre_op == "*":
                    stack.append(stack.pop()*num)
                elif pre_op == "/":
                    stack.append(int(stack.pop()/num))
                pre_op = each
                num = 0
        return sum(stack)

if __name__ == '__main__':
    s = Solution()
    print(s.calculate('3+2*2'))
    print(s.calculate('-5/2'))

