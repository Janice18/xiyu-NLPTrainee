"""
方法：栈
思路：用num保存上一个数字，用pre_op保存上一个操作符。当遇到新的操作符的时候，
需要根据pre_op进行操作。乘除的优先级高于加减。所以有以下规则：
之前的运算符是+，那么需要把之前的数字num进栈，然后等待下一个操作数的到来。
之前的运算符是-，那么需要把之前的数字求反-num进栈，然后等待下一个操作数的到来。
之前的运算符是×，那么需要立刻出栈和之前的数字相乘，重新进栈，然后等待下一个操作数的到来。
之前的运算符是/，那么需要立刻出栈和之前的数字相除，重新进栈，然后等待下一个操作数的到来。

注意比较的都是之前的操作符和操作数，现在遇到的操作符是没有什么用的。
"""
class Solution:
    def calculate(self, s: str) -> int:
        stack = []
        pre_op = '+'
        num = 0     #后面会用到，所以赋个初始值
        for i, each in enumerate(s):
            if each.isdigit():
                # 因为是一下子识别运算符前的数字的，比如识别到2位数2、6，则应该体现出2是十位
                #6是个位
                num = 10 * num + int(each)
            if i == len(s) - 1 or each in '+-*/':
                if pre_op == '+':
                    stack.append(num)
                elif pre_op == '-':
                    stack.append(-num)
                elif pre_op == '*':
                    stack.append(stack.pop() * num)
                elif pre_op == '/':
                    stack[-1] = int(stack[-1]/num)   #可能有负数，先用浮点数去除再取整
                pre_op = each
                num = 0
        return sum(stack)

if __name__ == '__main__':
    a1 = '3+2*2'
    a2 = '3/2'
    a3 = '3-5/2'
    print(Solution().calculate(a1))
    print(Solution().calculate(a2))
    print(Solution().calculate(a3))
