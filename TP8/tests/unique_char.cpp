#include "utils.hpp"
#include "timer.hpp"
#include <iostream>
#include <random>
#include <vector>
#include <cassert>

bool has_unique_chars(const char inSentence[], const size_t length){
    // TODO return true if all chars in inSentence are unique

    // LINEAR SPACE OCMPLEXITY
    if (length > 256)   // Must contain duplicate characters
        return false;
    else {

        bool allTheChars[256] = {0};                // Store characters in the string
        for (int i = 0; i < length; i++)            // Go throught he loop only once
        {
            if (allTheChars[int(inSentence[i])] == true)
                return false;

            allTheChars[int(inSentence[i])] = true;
        }
        return true;
    }


    // //NON  LINEAR SPACE OCMPLEXITY
    // bool unique = true;
    // for (size_t i = 0; i < length; ++i)
    //     {
    //     if(i < length-1){
                                             
    //         for (size_t j = i+1; j < length; ++j)
    //         {
    //             if (inSentence[j] == inSentence[i])
    //             {
    //                 unique = false;
    //                 break;
    //             }
    //         }
    //     }
    // }
    // return unique;
}

int main(int /*argc*/, char** /*argv*/){
    {
        CheckEqual(true, has_unique_chars("abc", 3));
    }
    {
        CheckEqual(false, has_unique_chars("abca", 4));
    }
    {
        CheckEqual(true, has_unique_chars("abcA", 4));
    }
    {
        CheckEqual(false, has_unique_chars("abc--", 5));
    }  
    
    return 0;
}