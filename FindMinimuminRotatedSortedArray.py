#FindMinimuminRotatedSortedArray.py

class Solution:
    # @param num, a list of integer
    # @return an integer
    def findMin(self,num):
        length = len(num)
        if num[0]<num[-1]:
            return num[0]
        else:
            left=0
            right=length-1
            while right-left>1:
                mid=(right+left)/2
                if num[left]<num[mid]:
                    left=mid
                else:
                    right=mid

            return min(num[left],num[right]) 
                        

if __name__ == "__main__":
    s=Solution()
    A=[0,1,2]
    r=s.findMin(A)
    print(r)
