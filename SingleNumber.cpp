/*

Given an array of integers, every element appears twice except for one. Find that single one.

Note:
Your algorithm should have a linear runtime complexity. Could you implement it without using extra memory?

*/

#include <iostream>

using namespace std;

class Solution {
public:
    int singleNumber(int A[], int n) {
        int x=A[0];
        for(int i=1;i<n;i++)        
            x ^=A[i];//异或可以将出现次数为偶数的清零
        return x;
    }
};

void main(){

Solution so;

A[7]={1,1,2,4,3,4,3};n=7;

x= so.singleNumber(A,n);

cout<<x<<endl;

}
