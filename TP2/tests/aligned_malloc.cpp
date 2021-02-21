#include <memory>
#include <cstdlib>
#include <iostream>
#include <cassert>

#include <cstddef>
#include "utils.hpp"

template <class ObjectType>
std::size_t alignementOfPtr(const ObjectType* ptr){
    if(ptr == nullptr){
        return 0;
    }

    // int biggestAlignment = alignof(std::max_align_t);        // Normal biggest aligment
    int biggestAlignment = 128;                	                // For strict(larger) alignment ==> over-alignment

    auto addPtr = reinterpret_cast<std::uintptr_t>(ptr);        // Properly convert pointer's address to integer
    
    int alignment = biggestAlignment;                           // Initial alignment value
    
    while (addPtr % alignment != 0){
        alignment /= 2;
    }
    
    return alignment;
}

void* custom_aligned_alloc(const std::size_t Alignment, const std::size_t inSize){
    if (inSize == 0) {
        return nullptr;
    }
    assert(Alignment != 0 && ((Alignment - 1) & Alignment) == 0);

    void * unalignedPtr = malloc(Alignment + inSize);               // Allocate more data than necessary
    size_t ptr = 0;                                                 // This "pointer" will be returned

    if (unalignedPtr != nullptr){
        /* First (naive) approach to finding the first aligned block - incrementation */
        // ptr = reinterpret_cast<size_t>(unalignedPtr);
        // while (alignementOfPtr(reinterpret_cast<void *>(ptr)) != Alignment)
        //     ptr += 1;

        /* Second (better) approach - binary arithmetics */
        ptr = (reinterpret_cast<size_t>(unalignedPtr) + size_t(Alignment) & ~size_t(Alignment-1));

        *(reinterpret_cast<void **>(ptr) - sizeof(void **)) = unalignedPtr;         // ptr's left neighbour points to the whole allocated data block
    }

    return reinterpret_cast<void *>(ptr);
}

void custom_free(void* ptr){
    if(ptr){
        void * unalignedPtr = *(reinterpret_cast<void **>(ptr) - sizeof(void **));  // The pointer left to ptr was created using malloc
        free(unalignedPtr);
    }
}

void test(){
    std::cout << "Test with hard numbers" << std::endl;
    {
        std::cout << "Address " << (1) << std::endl;
        std::cout << ">> Alignement " << alignementOfPtr(reinterpret_cast<unsigned char*>(1)) << std::endl;
        CheckEqual(1UL, alignementOfPtr(reinterpret_cast<unsigned char*>(1)));

        std::cout << "Address " << (2) << std::endl;
        std::cout << ">> Alignement " << alignementOfPtr(reinterpret_cast<unsigned char*>(2)) << std::endl;
        CheckEqual(2UL, alignementOfPtr(reinterpret_cast<unsigned char*>(2)));

        std::cout << "Address " << (4) << std::endl;
        std::cout << ">> Alignement " << alignementOfPtr(reinterpret_cast<unsigned char*>(4)) << std::endl;
        CheckEqual(4UL, alignementOfPtr(reinterpret_cast<unsigned char*>(4)));

        std::cout << "Address " << (8) << std::endl;
        std::cout << ">> Alignement " << alignementOfPtr(reinterpret_cast<unsigned char*>(8)) << std::endl;
        CheckEqual(8UL, alignementOfPtr(reinterpret_cast<unsigned char*>(8)));

        std::cout << "Address " << (6) << std::endl;
        std::cout << ">> Alignement " << alignementOfPtr(reinterpret_cast<unsigned char*>(6)) << std::endl;
        CheckEqual(2UL, alignementOfPtr(reinterpret_cast<unsigned char*>(6)));

        std::cout << "Address " << (7) << std::endl;
        std::cout << ">> Alignement " << alignementOfPtr(reinterpret_cast<unsigned char*>(7)) << std::endl;
        CheckEqual(1UL, alignementOfPtr(reinterpret_cast<unsigned char*>(7)));
    }
    std::cout << "Perform some allocations" << std::endl;                                   // QUESTION 6.1
    {
        const int nbAllocs = 10;
        for(int idx = 0 ; idx < nbAllocs ; ++idx){
            std::unique_ptr<int[]> ptr(new int[141234]);
            std::cout << "Address " << ptr.get() << std::endl;
            std::cout << ">> Alignement " << alignementOfPtr(ptr.get()) << std::endl;

            std::size_t alignement = alignementOfPtr(ptr.get());
            CheckEqual(true, (std::size_t(ptr.get()) & (alignement)) != 0);
            CheckEqual(true, (std::size_t(ptr.get()) & (alignement-1)) == 0);
        }
    }
    std::cout << "Test with C11" << std::endl;                                              // QUESTION 6.1
    {
        const int nbAllocs = 10;
        for(std::size_t alignment = 1 ; alignment <= 16 ; alignment *= 2){
            std::cout << "alignment = " << alignment << std::endl;
            for(int idx = 0 ; idx < nbAllocs ; ++idx){
                int* ptr = reinterpret_cast<int*>(aligned_alloc( alignment, sizeof(int)*141234));
                std::cout << "Address " << ptr << std::endl;
                std::cout << ">> Alignement " << alignementOfPtr(ptr) << std::endl;

                std::size_t alignement = alignementOfPtr(ptr);
                CheckEqual(true, (std::size_t(ptr) & (alignement)) != 0);
                CheckEqual(true, (std::size_t(ptr) & (alignement-1)) == 0);
                free(ptr);
            }
        }
    }
    std::cout << "Test with custom kernel" << std::endl;                                     // QUESTION 6.2
    {
        const int nbAllocs = 10;
        for(std::size_t alignment = 1 ; alignment <= 16 ; alignment *= 2){
            std::cout << "alignment = " << alignment << std::endl;
            for(int idx = 0 ; idx < nbAllocs ; ++idx){
                int* ptr = reinterpret_cast<int*>(custom_aligned_alloc( alignment, sizeof(int)*141234));
                std::cout << "Address " << ptr << std::endl;
                std::cout << ">> Alignement " << alignementOfPtr(ptr) << std::endl;

                std::size_t alignement = alignementOfPtr(ptr);
                CheckEqual(true, (std::size_t(ptr) & (alignement)) != 0);
                CheckEqual(true, (std::size_t(ptr) & (alignement-1)) == 0);
                custom_free(ptr);
            }
        }
    }
}


int main(){

    test();

    return 0;
}
