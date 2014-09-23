#include  <iostream>
#include <string>
using namespace std;

/*
Given a string, find the length of the longest substring without repeating characters. 
For example, the longest substring without repeating letters for "abcabcbb" is "abc", 
which the length is 3. For "bbbbb" the longest substring is "b", with the length of 1.

characters 是字符不是字母，所以leetCode测试上有
abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789! 一堆字符

ASCII能输入的字符上可以从032 空格开始，到127截至，所以我申请了96长度数组作为标志
IsExist：存储但前字符是否出现
position：记录字符出现的位置

我的思想可以理解为有两个指针，一个start指向字符串的开始字母，另一个其实就是但前i值。
通过IsExit来判断当前的s[i]是否出现：

如果没有出现，将但前字符所对应在IsExist元素置为true；

如果出现了，计算这个字符之前的字符到start长度，和maxLen比较。
再将这个字符前一次出现的位置的前一个字符到start指向的字符所对应的IsExit元素置为false
然后这个字符前一次出现的位置的后一个位置记为开始位置，将这个字符的位置更新


这样循环下去，最后一个没有重复字符出现的字符串长度不会计算，所以最后再做个比较
s.size()-start
就是最后一个字符串的长度
*/


class Solution {
public:
	int lengthOfLongestSubstring(string s) {
		int maxLen=0;
		int start=0;
		bool IsExist[96]={false};
		int position[96]={0};

		for (int i=0;i<s.size();i++)
		{
			if (IsExist[s[i]-32])
			{
				maxLen = maxLen > (i - start ) ? maxLen : (i - start);
				for(int j = position[s[i] - 32]-1; j >=start; j--) {
					IsExist[s[j] -32] = false;
				}		
				start = position[s[i] - 32] + 1;
				position[s[i] - 32] = i;
			}else{
				IsExist[s[i]-32]=true;
				position[s[i] - 32] = i;
			}
		}
		return maxLen>s.size()-start?maxLen:s.size()-start;
	}
};


void main(){

	string s="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!";
	cout<<s<<endl;
	Solution so;
	cout<<so.lengthOfLongestSubstring(s)<<endl;

}
