/*
There are N children standing in a line. Each child is assigned a rating value. 
You are giving candies to these children subjected to the following requirements: 

·Each child must have at least one candy. 
·Children with a higher rating get more candies than their neighbors. 

What is the minimum candies you must give? 

candy：分配糖果问题，一道一维动态规划的题目

解题思路一：

读懂题意可以判定这是一道一维动态规划题目，假设dp[i]代表第i个小孩发的糖果数：


当ratings[i] > ratings[i - 1]，必须比前一个多给一块，dp[i] = dp[i - 1] + 1
当ratings[i] == ratings[i - 1], 两个排名相同，则当前给一块即可，dp[i] = 1
当ratings[i] < ratings[i - 1], 排名比上一个低，应该比上一个少一块，dp[i] = 1;  
但是考虑特殊的case，当dp[i - 1] == 1时，这时设置dp[i]为1，然后依次递归，
for (int j = i - 1; j >= 0 && ratings[j] > ratings[j + 1]; j --) { dp[j] ++; }

这种方法时间复杂度是N*N ===>candy

解题思路二

不能思维僵化，再次题解题目，题目的要达到的要求是：

每个孩子都至少有一个糖果
具有较高等级的孩子们比他左右的孩子获得更多的糖果

因此根据题意，思路可以如下：

初始化数组dp，数组成员均为1 ，每个孩子先分配一个糖果
从左向右遍历，如果第i个孩子比第i - 1孩子等级高，则dp[i] = dp[i - 1] + 1
从右向左遍历，如果第i个孩子比第i + 1孩子等级高并且糖果比i+1糖果少，则dp[i] = dp[i + 1] + 1

==>candy1

*/

#include <iostream>
#include <vector>
using namespace std;

class Solution {
public:
    int candy(vector<int> &ratings) {
        
		int size = ratings.size();
		int i,j;
		int *num_candy = new int[size];
		num_candy[0]=1;
		for(i=1;i<size;i++){

			if(ratings.at(i)>ratings.at(i-1))
				num_candy[i]=num_candy[i-1]+1;
			else if(ratings.at(i)==ratings.at(i-1))
				num_candy[i]=1;
			else
				if(num_candy[i-1]>1)
					num_candy[i]=1;
				else{
					num_candy[i]=1;
					for(j=i-1;j>=0&&ratings[j]>ratings[j + 1]; j --) num_candy[j] ++;
				}
		}

		int num=0;

		for(i=0;i<size;i++) 
		{
			num+=num_candy[i];
		
		}
		
		return num;
    }


	    int candy1(vector<int> &ratings) {
        
		int size = ratings.size();
		int i;
		vector<int> num_candy(size,1);

		for(i=1;i<size;i++){
			if(ratings.at(i)>ratings.at(i-1))
				num_candy.at(i)=num_candy.at(i-1)+1;
		}

		for(i=size-2;i>=0;i--)
			if(ratings.at(i)>ratings.at(i+1)&&num_candy.at(i)<=num_candy.at(i+1))
				num_candy.at(i)=num_candy.at(i+1)+1;

		int num=0;

		for(i=0;i<size;i++) 
		{
			num+=num_candy.at(i);
		
		}
		
		return num;
    }

		int candy2(vector<int> &ratings) {//备忘录法，是一种递归方法，自顶向下的方法
		
			vector<int> f(ratings.size());
			int sum = 0;
			for (int i = 0; i < ratings.size(); ++i)
				sum += solve(ratings, f, i);
			return sum;
		}
		
		int solve(const vector<int>& ratings, vector<int>& f, int i){
			if (f[i] == 0) {
				f[i] = 1;
				if (i > 0 && ratings[i] > ratings[i - 1])
					f[i] = max(f[i], solve(ratings, f, i - 1) + 1);
				if (i < ratings.size() - 1 && ratings[i] > ratings[i + 1])
					f[i] = max(f[i], solve(ratings, f, i + 1) + 1);
			}
			return f[i];
		}
};

void main(){
	Solution so;

	vector<int> ratings(4);
	cout<<ratings.at(0);
	ratings.at(0)=1;
	ratings.at(1)=4;
	ratings.at(2)=3;
	ratings.at(3)=2;

	int min_candy_num=so.candy1(ratings);
	cout<<min_candy_num<<endl;

}
