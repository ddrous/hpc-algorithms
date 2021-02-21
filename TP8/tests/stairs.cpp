#include "utils.hpp"
#include "timer.hpp"
#include <iostream>
#include <random>
#include <vector>
#include <cassert>
#include <algorithm>
#include <stack>

int stairs(const int nbStairs){
    // TODO Compute the number of possibilities to climb N stairs/steps by 1, 2 or 3
    if (nbStairs==0){
        return 0;
    }
    else if (nbStairs==1){
        return 1;
    }
    else if (nbStairs==2){
        return 2;
    }
    else if (nbStairs==3){
        return 4;
    }

    else return stairs(nbStairs-1) + stairs(nbStairs-2) + stairs(nbStairs-3);

}

int main(int /*argc*/, char** /*argv*/){



    std::cout << "Small test" << std::endl;
    CheckEqual(0,stairs(0));
    CheckEqual(1,stairs(1));
    CheckEqual(2,stairs(2));
    CheckEqual(4,stairs(3));
    CheckEqual(7,stairs(4));
    CheckEqual(13,stairs(5));
    CheckEqual(24,stairs(6));
    CheckEqual(44,stairs(7));
    CheckEqual(81,stairs(8));
    CheckEqual(149,stairs(9));
    
    std::cout << "Large test" << std::endl;
    CheckEqual(0,stairs(0));
    CheckEqual(13,stairs(5));
    CheckEqual(274,stairs(10));
    CheckEqual(5768,stairs(15));
    CheckEqual(121415,stairs(20));
    CheckEqual(2555757,stairs(25));
    
    return 0;
}