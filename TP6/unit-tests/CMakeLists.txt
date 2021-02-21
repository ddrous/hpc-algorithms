###########################################################################
# SPETABARU - Berenger Bramas MPCDF - 2016
# Under LGPL Licence, please you must read the LICENCE file.
###########################################################################
project(UTESTS_CUDACPU CXX)

message("CUDACPU_CXX_FLAGS ${CUDACPU_CXX_FLAGS}")
ADD_DEFINITIONS(${CUDACPU_CXX_FLAGS})

# Find all code files (no GLOB_RECURSE)
file(
    GLOB	
    source_tests_files
    ./*.cpp
    )


# Adding the project sources dir as an include dir
INCLUDE_DIRECTORIES(
     ${CUDACPU_BINARY_DIR}/src    
     ${CUDACPU_SOURCE_DIR}/src    
     ${UTESTS_CUDACPU_SOURCE_DIR}
)


# Add execs - 1 cpp = 1 exec
foreach(exec ${source_tests_files}) 
    get_filename_component(
	    execname ${exec}
	    NAME_WE
    )
    if($ENV{VERBOSE})
        MESSAGE(STATUS "UTests -- Add ${execname}")
    endif()

    add_executable(
		    ${execname}
		    ${exec}
	    )

    target_link_libraries(
        ${execname}
        ${CUDACPU_LIBRARIES}
        )
    add_test(${execname} ${CMAKE_CURRENT_BINARY_DIR}/${execname})
endforeach(exec)
