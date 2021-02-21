project(CUDACPU CXX)
cmake_minimum_required(VERSION 3.0)

# Set a default build type if none was specified
set(default_build_type "Release")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/src")

set(CUDACPU_EXTRA_FLAG "-m64 -march=native -Wextra -Wnon-virtual-dtor -Wshadow -Wpointer-arith -Wcast-qual -Wconversion  -Wall -Wno-sign-conversion -pedantic -Woverloaded-virtual -Wpointer-arith -Wcast-qual -Wconversion -Wno-error" CACHE STRING "Additional flags"  )
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CUDACPU_EXTRA_FLAG}")

###########################################################

option(USE_SOLUTION "Should be ON to enable solutions" ON)

if(USE_SOLUTION)
    add_definitions(-DSOLUTION)
endif()

#################################################

option(USE_COVERAGE "Should be ON to enable code coverage" OFF)

if( USE_COVERAGE )
    message(STATUS "Use code coverage")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0 --coverage")
    list(APPEND CUDACPU_LIBRARIES gcov)
    # Then one can use these commande:
    # lcov --directory CMakeFiles/ -c -o coverage-rapport.info
    # genhtml -o ./coverage -t "Coverage example" coverage-rapport.info
endif()

#################################################

message(STATUS "CMAKE_CXX_FLAGS = ${CMAKE_CXX_FLAGS}")

file(GLOB_RECURSE cpp_for_lib ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
if(cpp_for_lib)
    file(GLOB_RECURSE hpp_for_lib ${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp)
    add_library(CUDACPU STATIC ${cpp_for_lib} ${hpp_for_lib})
    list(APPEND CUDACPU_LIBRARIES CUDACPU)
endif()

#################################################

find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

#################################################


file(GLOB_RECURSE cpp_for_examples ${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp)
foreach(an_example ${cpp_for_examples})
    get_filename_component(execname ${an_example} NAME_WE)
    
    message(STATUS "Add example ${execname} for ${an_example}")

    add_executable(${execname} ${an_example})
    if(CUDACPU_LIBRARIES)
        target_link_libraries(${execname} ${CUDACPU_LIBRARIES}) 
    endif()
endforeach()

#################################################
include(CTest)

option(USE_TESTING "Should be ON to enable unit tests" ON)

if( USE_TESTING )
    MESSAGE(STATUS "Testing is enable")
    set(CUDACPU_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    enable_testing()
    add_subdirectory(unit-tests)
endif()


