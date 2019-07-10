"""
方法一：哈希表
思路：因为单词长度固定的，我们可以计算出截取字符串的单词个数是否和 words 里相等，
所以我们可以借用哈希表。
一个哈希表是 words，一个哈希表是截取的字符串，比较两个哈希是否相等！
"""
class Solution:
    def findSubstring(self, s: str, words:[str]) ->[int]:
        from collections import Counter
        if not s or not words:return []
        one_word = len(words[0])
        all_len = len(words) * one_word
        n = len(s)
        wordsIndex = Counter(words)
        res = []
        for i in range(0, n - all_len + 1):
            tmp = s[i:i+all_len]
            c_tmp = []
            for j in range(0, all_len, one_word):
                c_tmp.append(tmp[j:j+one_word])
            if Counter(c_tmp) == wordsIndex:
                res.append(i)
        return res

"""
方法二：滑动窗口
思路：初始，left指针和right指针都指向S的第一个元素.
将 right指针右移，扩张窗口，直到得到一个可行窗口，亦即包含T的全部字母的窗口。
得到可行的窗口后，将left指针逐个右移，若得到的窗口依然可行，则更新最小窗口大小。
若窗口不再可行，则跳转至 2
"""
class Solution:
    def findSubstring(self, s: str, words: [str]) -> [int]:
        from collections import Counter
        if not s or not words:return []
        one_word = len(words[0])
        word_num = len(words)
        n = len(s)
        if n < one_word:return []
        wIndex = Counter(words)
        res = []
        for i in range(0, one_word):
            cur_cnt = 0
            left = i
            right = i
            cur_Counter = Counter()
            while right + one_word <= n:
                w = s[right:right + one_word]
                right += one_word
                if w not in wIndex:
                    left = right
                    cur_Counter.clear()
                    cur_cnt = 0
                else:
                    cur_Counter[w] += 1
                    cur_cnt += 1
                    while cur_Counter[w] > wIndex[w]:
                        left_w = s[left:left+one_word]
                        left += one_word
                        cur_Counter[left_w] -= 1
                        cur_cnt -= 1
                    if cur_cnt == word_num :
                        res.append(left)
        return res

if __name__ == '__main__':
    s1,words1 = 'barfoothefoobarman',['foo','bar']
    s2,words2 = 'wordgoodgoodgoodbestword',['word','good','best','word']
    print(Solution().findSubstring(s1,words1))
    print(Solution().findSubstring(s2,words2))

