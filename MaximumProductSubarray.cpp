#include <iostream>
#include <vector>
#include<math.h>

//  1）0，数组中有0，且其他部分都乘积比0小。这种情况可以把问题分解成一系列0隔开的子数组的最大乘积，再考虑上0.
//
//	2）非零，数组中有偶数个负数，则全乘起来比较大。
//
//	3）非零，数组中有奇数个负数，这时最大乘积有两种情况：最左边的负数+1到最右边，最左边到最右边负数-1。


using namespace std;

class Solution {
public:
	int maxProduct(int A[], int n) {
		
		if (n<1) return 0;
		if(n==1) return A[0];
	
		vector<int> Pos;
		vector<int> Neg;
		int maxTem=1;
		int MaxPro=0;
		int num=0;
		for (int i=0;i<n;i++)
		{
			if (A[i]!=0)
			{
				if (A[i]>0)
				{
					maxTem*=A[i];
			
				}
				else{					
					Pos.push_back(maxTem);
					Neg.push_back(A[i]);
					maxTem=1;			
				}
				num++;
			}	
			if (A[i]==0||i==n-1)
			{
				
				if (num==0||(num==1&&!Neg.empty()))//出现连续0和0与0之间只有一个负数的情况，这个子序列中的乘积应该为0
				{
					MaxPro = max(MaxPro,0);	
				}else{
					if (Neg.size()%2==0)//非零，数组中有偶数个负数，则全乘起来比较大。
					{			
						for (int j=0;j<Neg.size();j++)
						{
							maxTem=maxTem*Pos.at(j)*Neg.at(j);
						}
					}else{			//非零，数组中有奇数个负数，这时最大乘积有两种情况：最左边的负数+1到最右边，最左边到最右边负数-1。	
						int leftPro=Pos.at(0),rightPro=maxTem;
						for (int j=0;j<Neg.size()-1;j++)
						{
							leftPro=leftPro*Neg.at(j)*Pos.at(j+1);
							rightPro=rightPro*Neg.at(Neg.size()-1-j)*Pos.at(Neg.size()-1-j);
						}
						maxTem = max(leftPro,rightPro);
					}
					MaxPro = max(MaxPro,maxTem);
				}
				Pos.clear();
				Neg.clear();
				maxTem=1;
				num=0;
			}
		}

		return MaxPro;
	}
//简单版 ，计算当前位置的最大值和最小值，每次都要计算，
//一个正数乘以负数就变最小，负数乘负数就变最大，
//再将当前最大值和maxPro 全局最大值比较
	int maxProduct2(int A[], int n) {
		int pmax = A[0];
		int pmin = A[0];
		int maxPro = A[0];

		for (int i = 1; i < n; i++) {
			int tmax = A[i] * pmax;
			int tmin = A[i] * pmin;
			pmax = max(A[i], max(tmax, tmin));
			pmin = min(A[i], min(tmax, tmin));//当A中出现0的时候，tmax和tmin正好重新计算
			maxPro = max(pmax, maxPro);
		}
		return maxPro;
	}
};

void main(){

	Solution so;
	int A[]={-1,8,-2,0,78};

	cout<<so.maxProduct2(A,5)<<endl;

}
