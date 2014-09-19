# candy

class Solution:
    # @param ratings, a list of integer
    # @return an integer
    def candy(self, ratings):
        size = len(ratings)
        candy_count = [1]*size
        for i in range(1,size):
            if ratings[i]>ratings[i-1]:
                candy_count[i]=candy_count[i-1]+1
                print(candy_count)


        for j in range(0,size-1):
            i = size-j-2
            if ratings[i]>ratings[i+1] and candy_count[i]<=candy_count[i+1]:
                candy_count[i]=candy_count[i+1]+1
                print(candy_count)


        result = sum(candy_count)        
        return result
        


if __name__ == "__main__":
    s=Solution()
    ratings=[1,2,2]
    min_result=s.candy(ratings)
    print(min_result)
