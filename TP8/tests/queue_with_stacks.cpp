#include "utils.hpp"
#include "timer.hpp"
#include <iostream>
#include <cassert>
#include <stack>
// You should not include anything else here

class Queue {
    std::stack<int> s1;
    std::stack<int> s2;
    
public:
    void push(const int inVal){
    // TODO Add an element in the "queue"

        while (s1.size() != 0)      // put everything into s2
        {
            s2.push(s1.top());
            s1.pop();
        }
        
        s2.push(inVal);     // put inval into s2

        while (s2.size() != 0)      // put everything back into s1
        {
            s1.push(s2.top());
            s2.pop();
        }

    }
    
    int pop(){
    // TODO return the element at the top of the "queue" (and remove it)
        if (s1.size() == 0)
            std::cout << "This queue is empty!!" << "\n";
        else {
            int ret = s1.top();
            s1.pop();
            return ret;
        }
    }
    
    int size(){
    // TODO return the number of elements in the "queue"
        return s1.size() + s2.size();
    }
};

int main(int /*argc*/, char** /*argv*/){
    {
        Queue q;
        CheckEqual(0, q.size());
        q.push(1);
        CheckEqual(1, q.size());
        q.push(2);
        CheckEqual(2, q.size());
        CheckEqual(1, q.pop());
        CheckEqual(2, q.pop());
    }
    
    {
        Queue q;
        CheckEqual(0, q.size());
        q.push(1);
        CheckEqual(1, q.size());
        q.push(2);
        CheckEqual(2, q.size());
        CheckEqual(1, q.pop());        
        q.push(3);
        CheckEqual(2, q.size());
        CheckEqual(2, q.pop());
        CheckEqual(3, q.pop());
    }
    
    Queue q;
    for(int idx = 0 ; idx < 100 ; ++idx){
        q.push(idx);
        CheckEqual(idx+1, q.size());    
    }
    for(int idx = 0 ; idx < 100 ; ++idx){
        CheckEqual(idx, q.pop());
        CheckEqual(100-idx-1, q.size());    
    }
    
    return 0;
}