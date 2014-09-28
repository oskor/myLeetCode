# LengthOfLongestSubstring.py
#利用字典

class Solution:
    # @return an integer
    def lengthOfLongestSubstring(self, s):
        D={}
        start=0
        maxLen=0;
        for i in range(0,len(s)):
            if s[i] in D:
                maxLen=max(maxLen,i-start)
                for j in range(start,D[s[i]]):
                    D.pop(s[j])
                start=D[s[i]]+1
            D[s[i]]=i

        return max(maxLen,len(s)-start)


if __name__ == '__main__':
    s=Solution()
    string='abcabcbb'
    maxLen=s.lengthOfLongestSubstring(string)
    print(maxLen)
