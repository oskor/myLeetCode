#include <iostream>
#include <string>
#include <sstream>
using namespace std;

class Solution {
public:
    void reverseWords(string &s) {
		//int s_length = s.length();
		//string s_temp(s_length,'0');
		//int i,count =0,j;
		//int flag_firstnonspace=1;
		//for(i=s_length-1;i>=0;){	

		//	if(s[i]!= ' '&&flag_firstnonspace==1){
		//		j=i;
		//		flag_firstnonspace=0;
		//	}
		//	i--;
		//	if(i<0&&flag_firstnonspace==1)break;

		//	if((i<0&&flag_firstnonspace==0)||(s[i]== ' '&&s[i+1]!=' ')){
		//		
		//		for(int k =i+1;k<=j;k++)
		//		{
		//			s_temp[count]=s[k];
		//			count++;
		//		}
		//		if(count>=s_length){count++;break;}
		//		 s_temp[count]=' ';
		//		 count++;
		//		 flag_firstnonspace=1;
		//	}
		//}
		//
		//for(i=0;i<count-1;i++)
		//	s[i]=s_temp[i];
		//s.erase(i,s_length-i+1);

/*======================简单版本================*/
		  istringstream is(s);
          string tmp="";
          string out="";
          while(is>>tmp){
				cout<<tmp<<endl;
                tmp+=" ";
                tmp+=out;
                out=tmp;
				cout<<out<<endl;
           }
           s=out.substr(0,out.length()-1);
    }
};

void main()
{
	string s="   the sky   is blue  ";
	cout<<s<<s.length()<<endl;
	Solution so;
	so.reverseWords(s);
	cout<<s<<s.length()<<endl;


}
