project(PRACTICAL_WORK CXX)
cmake_minimum_required(VERSION 3.0)

###########################################################

set(PRACTICAL_WORK_EXTRA_FLAG "-m64 -march=native -Wextra -Wnon-virtual-dtor -Wshadow -Wpointer-arith -Wcast-qual -Wconversion  -Wall -Wno-sign-conversion -pedantic -Woverloaded-virtual -Wpointer-arith -Wcast-qual -Wconversion -Wno-error" CACHE STRING "Additional flags"  )
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${PRACTICAL_WORK_EXTRA_FLAG}")

message(STATUS "CMAKE_CXX_FLAGS = ${CMAKE_CXX_FLAGS}")

###########################################################

option(USE_SOLUTION "Should be ON to enable solutions" ON)

if(USE_SOLUTION)
    add_definitions(-DSOLUTION)
endif()

###########################################################

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/src")

###########################################################

file(GLOB_RECURSE cpp_for_lib ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
if(cpp_for_lib)
    file(GLOB_RECURSE hpp_for_lib ${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp)
    add_library(pratical_work_core STATIC ${cpp_for_lib} ${hpp_for_lib})
    list(APPEND PRACTICAL_WORK_CORE_LIBRARIES pratical_work_core)
endif()

###########################################################

option(USE_TESTING "Should be ON to enable unit tests" ON)

if(USE_TESTING)
    enable_testing()
endif()

###########################################################

file(GLOB_RECURSE test_files ${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp)
foreach(test_file ${test_files})
    get_filename_component(execname ${test_file} NAME_WE)
    
    message(STATUS "Add test ${execname} for ${test_file}")

    add_executable(${execname} ${test_file})
    if(PRACTICAL_WORK_CORE_LIBRARIES)
        target_link_libraries(${execname} ${PRACTICAL_WORK_CORE_LIBRARIES}) 
    endif()
    
    if(USE_TESTING)
        add_test(${execname} ${CMAKE_CURRENT_BINARY_DIR}/${execname})
    endif()
endforeach()


