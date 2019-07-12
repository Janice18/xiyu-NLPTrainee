class Solution:
    def minWindow(self, s:str, t:str) ->str:
        if len(s) < len(t):return ''
        T = {}    #存储t中字母及其个数
        for i in t:
            if i in T:
                T[i] +=1
            else:
                T[i] = 1

        l, r = 0, 0  #左右指针初始化
        count = 0    #计数s中出现t中字母的个数
        ans = ''
        minL = len(s)+1
        while r < len(s):
            if s[r] in T:
                T[s[r]] -= 1
                if T[s[r]] >= 0:
                    count += 1
                while count == len(t):
                    if (r-l+1)<minL:
                        minL = r-1+1
                        ans = s[l:r+1]
                    if s[l] in T:
                        T[s[l]] +=1
                        if T[s[l]] > 0:
                            count -=1
                    l += 1
            r += 1
        return ans

if __name__ == '__main__':
    s, t = 'ADOBECODEBANC', 'ABC'
    s1,t1 = 'ABCDEFG', 'ABCDEFG'
    print(Solution().minWindow(s, t))
    print(Solution().minWindow(s1, t1))


