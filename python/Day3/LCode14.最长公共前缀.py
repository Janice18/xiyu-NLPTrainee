"""
思路：第一步：找出最短的字符串
第二步：把第一个字符串默认为最大，依此与长度最短的字符串比较
"""

class Solution(object):
    def longestSubFix(self,strs):
        if not strs:
            return ""
        if len(strs) == 1:
            return strs[0]
        minl = min([len(x) for x in strs])
        end = 0
        while end < minl:
            for i in range(1, len(strs)):
                if strs[i][end] != strs[i - 1][end]:
                    return strs[0][:end]
            end += 1
        return strs[0][:end]


def main():
    a = ['flower', 'flow', 'flight']
    b = ['dog', 'racecar', 'car']
    print(Solution().longestSubFix(a))
    print(Solution().longestSubFix(b))

if __name__ == '__main__':
    main()


class Solution(object):
    def longestCommonPrefix(self, strs):
        """
        :type strs: List[str]
        :rtype: str
        """
        # 判断是否为空
        if not strs:
            return ''
        # 在使用max和min的时候已经把字符串比较了一遍
        # 当前列表的字符串中，每个字符串从第一个字母往后比较直至出现ASCII码 最小的字符串
        s1 = min(strs)
        # 当前列表的字符串中，每个字符串从第一个字母往后比较直至出现ASCII码 最大的字符串
        s2 = max(strs)
        # 使用枚举变量s1字符串的每个字母和下标
        for i, c in enumerate(s1):
            # 比较是否相同的字符串，不相同则使用下标截取字符串
            if c != s2[i]:
                return s1[:i]
        return s1


if __name__ == '__main__':
    s = Solution()
    print(s.longestCommonPrefix(["flower", "flow", "flight"]))
    print('123', s.longestCommonPrefix(["dog", "racecar", "car"]))

