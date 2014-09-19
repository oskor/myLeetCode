#include <iostream>
#include <string>
#include <vector>
#include <sstream>

using namespace std;

class Solution {
public:
    int evalRPN(vector<string> &tokens) {
      
		int result,temp;
		int vector_size = tokens.size();
		int i;
		vector<int> storage;
		for(i=0;i<vector_size;i++){
		
			
			if((tokens.at(i)[0]!='/')&&(tokens.at(i)[0]!='*')&&(tokens.at(i)[0]!='+')&&(tokens.at(i)[0]!='-'||(tokens.at(i)[0]=='-'&&tokens.at(i).length()!=1))){		
				stringstream ss;
				ss<<tokens.at(i);
				ss>>temp;
				storage.push_back(temp);			
			}else{
			    int a1,a2;
				switch(tokens.at(i)[0]){			
				case '+':a2=storage.back();storage.pop_back();a1=storage.back();storage.pop_back(); result = a1+a2;storage.push_back(result);break;
				case '-':a2=storage.back();storage.pop_back();a1=storage.back();storage.pop_back(); result = a1-a2;storage.push_back(result);break;
				case '*':a2=storage.back();storage.pop_back();a1=storage.back();storage.pop_back(); result = a1*a2;storage.push_back(result);break;
				case '/':a2=storage.back();storage.pop_back();a1=storage.back();storage.pop_back(); result = a1/a2;storage.push_back(result);break;
				}
			}
		}

		result = storage.back();
		return result;
    }
};


int main(){

	//"2", "1", "+", "3", "*"
	vector<string> tokens(1);
	tokens.at(0)="3";
	//tokens.at(1)="-4";
	//tokens.at(2)="+";
/*	tokens.at(3)="3";
	tokens.at(4)="/"*/;

	Solution so;
	int result=so.evalRPN(tokens); 
	cout<<result<<endl;
	
	return 0;

}
