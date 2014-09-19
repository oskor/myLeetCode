# EvaluateReversePolishNotation


class Solution:
    # @param tokens, a list of string
    # @return an integer
    def evalRPN(self, tokens):
        storage=[]
        size = len(tokens)
        for i in range(0,size):
            if tokens[i] in ['+','*','/','-']:
                a2 = storage[-1]
                storage.pop(-1)
                a1 = storage[-1]
                storage.pop(-1)
                if tokens[i][0] == '+':                   
                    result = a1+a2
                    storage.append(result)
                elif tokens[i][0] == '-':
                    result = a1-a2
                    storage.append(result)
                elif tokens[i][0] == '*':
                    result = a1*a2
                    storage.append(result)
                else:
                    if a1%a2 != 0 and a1/a2< 0:
                        result = a1/a2+1
                    else:
                        result = a1/a2
                    storage.append(int(result))             
            else:
                storage.append(int(tokens[i]))
        else:
            result = storage[0]

        return result
        

if __name__ == "__main__":
    s=Solution()
    tokens=["3","-4","+"]
    t=s.evalRPN(tokens)
    print(t)


