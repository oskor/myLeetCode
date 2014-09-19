# single number

class Solution:
    # @param A, a list of integer
    # @return an integer
    def singleNumber(self, A):
        x = A[0]
        for i in range(1,len(A)):
            x = x^A[i]

        return x



if __name__ == "__main__":
    s=Solution()
    A=[1,1,2,3,4,5,4,3,5]
    r=s.singleNumber(A)
    print(r)
