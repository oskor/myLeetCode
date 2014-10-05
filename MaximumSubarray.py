#MaximumSubarray.py
class Solution:
    # @param A, a list of integers
    # @return an integer
    def maxSubarray(self, A):
        pmax=A[0]
        pmin=A[0]
        maxPro=A[0]
        for i in range(1,len(A)):
            tmax = pmax+A[i]
            pmax = max(A[i],tmax)
            maxPro=max(maxPro,pmax)

        return maxPro


if __name__ == '__main__':
    s=Solution()
    A=[-1,-2,-1,-3]
    result=s.maxSubarray(A)
    print(result)
