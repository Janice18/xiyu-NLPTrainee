# """
# 法一：使用字典
# 思路：建立两个字典rDict，mDict，其中key-value分别为字母的值和出现的次数
# 使用字典的get方法，通过设置第二个参数使得未寻找的情况下返回0。
# """
# class Solution:
#     def canConstruct(self, ransomNote: 'str', magazine: 'str') -> 'bool':
#         rDict, mDict = dict(),dict()
#         for ri in ransomNote:
#             rDict[ri] = rDict.get(ri, 0) + 1
#         for mi in magazine:
#             mDict[mi] = mDict.get(mi, 0) + 1
#
#         for ri in rDict.keys():
#             if ri in mDict and rDict[ri] <=  mDict[ri]:
#                 continue
#             else:
#                 return  False
#         return  True


"""
法二：适用Counter
思路：对字典的建立可以进一步优化，直接使用collections模块中的Counter类，用于追踪值的出现次数
（Counter类是对字典的补充）。其次，只做一次字典的建立，即对magazine字符串使用Counter统计单词
出现的次数。单独遍历ransom字符串的每一个字符进行比较处理。
"""
# class Solution:
#     def canConstruct(self, ransomNote: 'str', magazine: 'str') -> 'bool':
#         from collections import Counter
#         mDict = Counter(magazine)
#
#         for ri in ransomNote:
#             if ri in mDict and mDict[ri] > 0:
#                 mDict[ri] -= 1
#             else:
#                 return  False
#         return True


"""
法三：使用字符串或列表的方法
思路：对magazine字符串进行列表化，使用列表的remove方法进行筛选。若不存在，
则抛出异常被try-except结构检测到执行except中的语句
"""
# class Solution:
#     def canConstruct(self, ransomNote: 'str', magazine: 'str') -> 'bool':
#         mlist = list(magazine)
#
#         try:
#             for ri in ransomNote:
#                 mlist.remove(ri)
#             return True
#         except:
#             return False

"""
法四：省去所有的建立过程，直接对magazine字符串使用字符串的replace方法,暂时有错误
"""
class Solution:
    def canConstruct(self, ransomNote: 'str', magazine: 'str') -> 'bool':
        for ri in ransomNote:
            if ri in magazine:
                magazine.replace(ri, '', 1)
            else:
                return False
        return True


if __name__ == '__main__':
    s = Solution()
    print(s.canConstruct('a','b'))
    print(s.canConstruct('aa','ab'))
    print(s.canConstruct('aa', 'aab'))






