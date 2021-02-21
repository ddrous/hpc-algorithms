#ifndef UTILS_HPP
#define UTILS_HPP

#include <cmath>
#include <iostream>

void CheckEqualCore(const float f1, const float f2, const char* file, const int line){
    if((f1 != 0 && std::abs(f1-f2)/std::abs(f1) > 1E-7)
        || (f1 == 0 && std::abs(f2) > 1E-7)){
        std::cout << "Error!" << std::endl;
        std::cout << "File: " << file << std::endl;
        std::cout << "Line: " << line << std::endl;
        std::cout << "Should be: " << f1 << std::endl;
        std::cout << "Is: " << f2 << std::endl;
        exit(1);
    }
}

void CheckEqualCore(const double f1, const double f2, const char* file, const int line){
    if((f1 != 0 && std::abs(f1-f2)/std::abs(f1) > 1E-12)
        || (f1 == 0 && std::abs(f2) > 1E-12)){
        std::cout << "Error!" << std::endl;
        std::cout << "File: " << file << std::endl;
        std::cout << "Line: " << line << std::endl;
        std::cout << "Should be: " << f1 << std::endl;
        std::cout << "Is: " << f2 << std::endl;
        exit(1);
    }
}

template <class NumType>
void CheckEqualCore(const NumType v1, const NumType v2, const char* file, const int line){
    if(v1 != v2){
        std::cout << "Error!" << std::endl;
        std::cout << "File: " << file << std::endl;
        std::cout << "Line: " << line << std::endl;
        std::cout << "Should be: " << v1 << std::endl;
        std::cout << "Is: " << v2 << std::endl;
        exit(1);
    }
}

#define CheckEqual(GOODV, BADV) CheckEqualCore((GOODV), (BADV), __FILE__, __LINE__)


#if defined(__AVX2__)

#include <tmmintrin.h>
#include <emmintrin.h>
#include <immintrin.h>

void printVec(const __m256d vec){
    double values[4];
    _mm256_storeu_pd(&values[0], vec);
    std::cout << "Vec AVX : " << values[0] << " " << values[1]
              << " " << values[2] << " " << values[3] << std::endl;
}

#endif

#if defined(__SSE3__)

#include <emmintrin.h>
#include <pmmintrin.h>

void printVec(const __m128d vec){
    double values[2];
    _mm_storeu_pd(&values[0], vec);
    std::cout << "Vec SSE: " << values[0] << " " << values[1] << std::endl;
}

#endif

#endif