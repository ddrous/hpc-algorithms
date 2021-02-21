#include "UTester.hpp"

#include "cudacpu_extra.h"

class TestIdxToPos : public UTester< TestIdxToPos > {
    using Parent = UTester< TestIdxToPos >;
    
    void TestBasic() {
        {
            const dim3 size(200,88,100);
            const int x = 150;
            const int y = 6;
            const int z = 99;
            const int idx = (x * size.y + y) * size.z + z;
            const int3 pos = dim3_1DidxToPos(idx, size);
            UASSERTEEQUAL(x, pos.x);
            UASSERTEEQUAL(y, pos.y);
            UASSERTEEQUAL(z, pos.z);
        }
    }

    void TestLoop() {
        const dim3 size(200,88,100);
        for(int x = 0 ; x < size.x ; ++x){
            for(int y = 0 ; y < size.y ; ++y){
                for(int z = 0 ; z < size.z ; ++z){
                    const int idx = (x * size.y + y) * size.z + z;
                    const int3 pos = dim3_1DidxToPos(idx, size);
                    UASSERTEEQUAL(x, pos.x);
                    UASSERTEEQUAL(y, pos.y);
                    UASSERTEEQUAL(z, pos.z);
                }
            }
        }
    }

    void SetTests() {
        Parent::AddTest(&TestIdxToPos::TestBasic, "Basic test for dim3_1DidxToPos");
        Parent::AddTest(&TestIdxToPos::TestLoop, "Loop test for dim3_1DidxToPos");
    }
};

// You must do this
TestClass(TestIdxToPos)