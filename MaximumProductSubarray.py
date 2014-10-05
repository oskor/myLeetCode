#MaximumProductSubarray.py
class Solution:
    # @param A, a list of integers
    # @return an integer
    def maxProduct(self, A):
        pmax=A[0]
        pmin=A[0]
        maxPro=A[0]
        for i in range(1,len(A)):
            tmax = pmax*A[i]
            tmin = pmin*A[i]
            pmax = max(A[i],max(tmax,tmin))
            pmin = min(A[i],min(tmax,tmin))
            maxPro=max(maxPro,pmax)

        return maxPro


if __name__ == '__main__':
    s=Solution()
    A=[1,2,-1,3]
    result=s.maxProduct(A)
    print(result)
