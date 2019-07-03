"""
方法：递归
思路：共两步
第一步：挑选出所有可能出现在第一个位置的字符（即把第一个字符和后面的字符交换）
第二步：固定第一个字符，求后面所有字符的排列（继续把后面字符分为前述两部分，然后递归处理）
注意:考虑去重复（集合），考虑字典序打印（排序）
"""
class Solution:
    def Permutation(self, s):
        if not s:
            return []
        arr = []
        self.backtrack(s, arr, '')
        return sorted(list(set(arr)))

    def backtrack(self, s, arr, path):     #其中path用来存储字符串的排列组合
        if not s:
            arr.append(path)
        else:
            for i in range(len(s)):
                self.backtrack(s[:i] + s[i+1:], arr, path + s[i])   


print(Solution().Permutation('acdhs'))

#表述方法2，
class Solution:
    def Permutation(self, ss):
        out = []
        if len(ss) == 0:
            return out
        charlist = list(ss)
        self.permutation1(charlist, 0, out)
        out = [''.join(out[i]) for i in range(len(out))]    #列表转化为字符串
        out.sort()
        return out

    def permutation1(self, ss, begin, out):
        if begin == len(ss)-1:
            out.append(ss[:])
        else:
            for i in range(begin, len(ss)):
                # 如果是重复字符，跳过
                if ss[begin] == ss[i] and begin != i:
                    continue
                else:
                    # 依次与后面每个字符交换
                    ss[begin], ss[i] = ss[i], ss[begin]
                    self.permutation(ss, begin + 1, out)
                    # 回到上一个状态
                    ss[begin], ss[i] = ss[i], ss[begin]


print(Solution().Permutation('abc'))

