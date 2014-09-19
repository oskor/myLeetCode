# reverse words

class Solution:
    # @param s, a string
    # @return a string
    def reverseWords(self, s):
        sep=s.split()
        sep.reverse()
        s=' '.join(sep)
        return s

        

if __name__ == "__main__":
    s=Solution()
    t=s.reverseWords(" aa aa ")
    print t
