"""
方法：滑动窗口
思路：初始，left指针和right指针都指向S的第一个元素.
将 right 指针右移，扩张窗口，直到得到一个可行窗口，亦即包含T的全部字母的窗口。
得到可行的窗口后，将left指针逐个右移，若得到的窗口依然可行，则更新最小窗口大小。
若窗口不再可行，则跳转至 2
"""
class Solution:
    def minWindow(self,s:str, t:str) -> str:
        if len(s) < len(t):
            return ''
        T = {}
        for i in t:
            if i in t:
                if i in T:
                    T[i] += 1
                else:
                    T[i] = 1
        l = 0
        r = 0
        count = 0  #s中出现t中元素的个数
        ans = ''
        minl = len(s) + 1  # 取初始最小长度为len(s)+1，因为可能s,t同长度
        while r < len(s):
            if s[r] in T:
                T[s[r]] -= 1
                if T[s[r]] >= 0:  # 表示是必须字符，若<0，表明该字符多了
                    count += 1
                while count == len(t):  # 计数符合要求，r移动完毕，同时准备移动l
                    if (r - l + 1) < minl:  # 测量长度是否符合最小
                        ans = s[l:r + 1]
                        minl = r - l + 1  # 更新最小长度
                    if s[l] in T:  # 因为l向右移动，s[l]都要移出刚才得到的字符串
                        T[s[l]] += 1
                        if T[s[l]] > 0:  # 若移出的字符使得字符串不满足要求，count-1退出循环
                            count -= 1
                    l = l + 1  # 继续移动l，即继续删除字符，直到不满足为止
            r = r + 1
        return ans

if __name__ == '__main__':
    s, t = 'ADOBECODEBANC', 'ABC'
    print(Solution().minWindow(s, t))


