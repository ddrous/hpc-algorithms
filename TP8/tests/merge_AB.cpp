#include "utils.hpp"
#include "timer.hpp"
#include <iostream>
#include <random>
#include <vector>
#include <cassert>
#include <algorithm>

void merge_AB(int A[], const int lengthA,
              const int B[], const int lengthB){
    // TODO A and B are sorted
    // TODO merge B into A while keeping all values sorted
    // TODO allocated size of A is at least (lengthA + lengthB)

    // for (int j = 0; j < lengthA; j++){ 
    //         std::cout << A[j] << "  ";
    // }

    if (lengthA != 0 && lengthB !=0){


        //////////// NON-LINEAR VERSION ///////////////////////////////////////

        // std::cout<< "Early LenA " << lengthA << "\n";
        // int currlengthA = lengthA;

        // for (int j = 0; j < lengthB; j++){      // Parcours de B

        //     int i = currlengthA-1;
        //     while (A[i] > B[j])
        //     {
        //         i -= 1;
        //     }
            
        //     std::cout<< "i " << i << "  " << A[i] << "\n";
                
        //     currlengthA += 1;

        //     // if (B[j] < A[i]){
        //         // for (int k = currlengthA-1; k >= i; k--){   // shift values in A to the right
        //         //     A[k+1] = A[k];
        //         // }

        //     // for (int k = i; k < currlengthA-1; k++){   // shift values in A to the right
        //     for (int k = currlengthA; k > i; k--){
        //         A[k] = A[k-1];
        //         std::cout<< "A[k+1] "  << A[k+1] << "\n";
        //     }
        //     A[i] = B[j];

        //     // }

        // }
        
        // std::cout<< "Fianl LenA " << currlengthA << "\n";
        //////////// ///////////////////////////////////////



        //////////// LINEAR VERSION ///////////////////////////////////////

        // Let's move all elements at the end of A
        int j = lengthA+lengthB-1;
        for (int i = lengthA-1; i >= 0; i--){ 
                A[j] = A[i];
                j--;
        }

        int i = lengthB;        // index dans A
        j = 0;              // index dans B
        int k = 0;              // index dans A+B

        while (k < lengthA+lengthB)
        {
            /* take el in A*/
            if ((A[i] <= B[j] && i < (lengthA+lengthB)) || (j == lengthB)){
                A[k] = A[i];
                k += 1;
                i += 1;
            }
            /* take el in B*/
            else{
                A[k] = B[j];
                k += 1;
                j += 1;
            } 
        }
        

    }
        ////////////////////////////////////////////////////////

}

int main(int /*argc*/, char** /*argv*/){
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 10);
            
    for(int idxSizeA = 1 ; idxSizeA < 100 ; ++idxSizeA){
        for(int idxSizeB = 1 ; idxSizeB < 100 ; ++idxSizeB){
            std::vector<int> A(idxSizeA);
            std::vector<int> B(idxSizeB);
        

            for(int idxA = 0 ; idxA < idxSizeA ; ++idxA){
                if(idxA == 0){
                    A[idxA] = dis(gen);
                }
                else{
                    A[idxA] = dis(gen) + A[idxA-1];
                }
            }
            
            for(int idxB = 0 ; idxB < idxSizeB ; ++idxB){
                if(idxB == 0){
                    B[idxB] = dis(gen);
                }
                else{
                    B[idxB] = dis(gen) + B[idxB-1];
                }
            }
        
            // Not a good solution, merge the array, and sort them....
            std::vector<int> AB = A;
            AB.insert(AB.end(), B.begin(), B.end());
            std::sort(AB.begin(), AB.end());
        
            // A[0]=0;
            // A[1]=1;
            // A[2]=1;
            // A[3]=2;
            // A[4]=3;
            // A[5]=4;
            // A[6]=4;


            // B[0]=0;
            // B[1]=0;
            // B[2]=4;
            // B[3]=4;
            // B[4]=5;

            A.resize(idxSizeA + idxSizeB);
            merge_AB(A.data(), idxSizeA, B.data(), idxSizeB);
            CheckEqual(true, AB == A);
        }
    }
    
    return 0;
}