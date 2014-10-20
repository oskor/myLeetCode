#include <iostream>
#include <vector>
using namespace std;

//Suppose a sorted ( from min to max)array is rotated at some pivot unknown to you beforehand.
//
//	(i.e., 0 1 2 4 5 6 7 might become 4 5 6 7 0 1 2).
//
//	Find the minimum element.
//
//	You may assume no duplicate exists in the array.

class Solution {
public:
	int findMin(vector<int> &num) {
		int len = num.size();
		int min_e;

		if (num.at(0)<num.at(len-1))
		{
			min_e=num.at(0);
		}else{
			int left=0,right=len-1;//二分法
			while(right-left>1){
				//cout<<left<<'\t'<<right<<endl;
				int mid = (left+right)/2;
				if (num.at(mid)>num.at(left))
				{
					left=mid;
				}else{
					right=mid;
				}
			}
			min_e = min(num.at(left),num.at(right));
		}
		return min_e;
	}
};

void main(){

	Solution so;

	vector<int> num(8);

	num.at(0)=4;num.at(1)=5;num.at(2)=6;num.at(3)=7;num.at(4)=0;num.at(5)=1;num.at(6)=2;num.at(7)=3;

	int min_e = so.findMin(num);

	cout<<min_e;

}
