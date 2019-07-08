"""
方法：回溯法+DFS
思路：
IP地址由四部分构成，可以设置一个变量segment,当segment = 4时，可结束循环，将结果添加到列表中；
每个部分数值均值0---255之间，因此每次回溯最多需要判断3个元素，即当前元素i---i+2这三位数字。
"""
class Solution(object):
    def restoreIpAddresses(self, s):
        """
        :type s: str
        :rtype: List[str]
        """
        res = []

        def dfs(s, segment, res, ip):
            if segment == 4:
                if s == '':
                    res.append(ip[1:])
                return
            for i in range(1,4):
                if i <= len(s):
                    if int(s[:i]) <= 255:
                        dfs(s[i:],segment+1,res,ip+'.'+s[:i])
                        if s[0] == '0':
                            break


        dfs(s, 0, res, '')  # segment 初始化为0
        return res


if __name__ == '__main__':
    S= Solution()
    s= "25525511135"
    print(S.restoreIpAddresses(s))


class Solution:
    def restoreIpAddresses(self, s):
        """
        :type s: str
        :rtype: list[str]
        """
        res = []
        self.dfs(s, [], res)
        return res

    def dfs(self, s, segment, res):
        if len(s) > (4 - len(segment)) * 3:
            return
        if not s and len(segment) == 4:
            res.append('.'.join(segment))
            return
        for i in range(min(3, len(s))):
            curr = s[:i + 1]
            if (curr[0] == '0' and len(curr) >= 2) or int(curr) > 255:
                continue
            self.dfs(s[i + 1:], segment + [s[:i + 1]], res)



if __name__ == '__main__':
    print(Solution().restoreIpAddresses('25525511135'))


class Solution:
    def restoreIpAddresses(self, s: str) -> list:
        res = []
        n = len(s)

        def backtrack(i, tmp, flag):
            if i == n and flag == 0:
                res.append(tmp[:-1])
                return
            if flag < 0:
                return
            for j in range(i, i + 3):
                if j < n:
                    if i == j and s[j] == "0":
                        backtrack(j + 1, tmp + s[j] + ".", flag - 1)
                        break
                    if 0 < int(s[i:j + 1]) <= 255:
                        backtrack(j + 1, tmp + s[i:j + 1] + ".", flag - 1)

        backtrack(0, "", 4)
        return res




