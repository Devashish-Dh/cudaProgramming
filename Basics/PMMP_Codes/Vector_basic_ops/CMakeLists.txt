CMAKE_MINIMUM_REQUIRED ( VERSION 3.18)

PROJECT(Chapter_2_Codes
        VERSION 1.0
        DESCRIPTION "Contains codes and exercises from chapter 2 of the PMMP book." 
        LANGUAGES CXX CUDA       
)

ENABLE_LANGUAGE ( CUDA )

set(CMAKE_BUILD_TYPE DEBUG)

SET ( CMAKE_CXX_STANDARD 17 )
SET ( CMAKE_CUDA_STANDARD 17 )


SET ( SOURCES 
    ${CMAKE_SOURCE_DIR}/src
)

ADD_EXECUTABLE ( # remember to include the .cpp and .cu files here 

vector_addition ${SOURCES}/vector_addition.cu 

)

INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/include ) # keep all header files here, .c .cpp .cuh

#TARGET_LINK_LIBRARIES( ${CMAKE_PROJECT_NAME}
#    PRIVATE 

TARGET_COMPILE_OPTIONS( 

    vector_addition


    PRIVATE $<$<COMPILE_LANGUAGE:CXX>: > # add flags etc for compilers here later when writing codes for release / debug etc 
    PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: > 
    )


#TARGET_SOURCES( ${CMAKE_PROJECT_NAME} PRIVATE
#    src/____write_.cu/.cpp_files_here____
#)

SET_PROPERTY( TARGET  
vector_addition

PROPERTY CUDA_SEPARABLE_COMPILATION ON )

SET_PROPERTY( TARGET  
vector_addition

PROPERTY CUDA_ARCHITECTURES 61 62 70 72 75 )
