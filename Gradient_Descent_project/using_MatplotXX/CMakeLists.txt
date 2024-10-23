CMAKE_MINIMUM_REQUIRED(VERSION 3.5)

PROJECT(exploring_matplot++
        VERSION 1.0
        DESCRIPTION "simple experiments" 
        LANGUAGES CXX       
)

INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/include) # for header files if any

SET(SOURCES ${CMAKE_SOURCE_DIR}/src/)# source files of the project







ADD_EXECUTABLE(Project_Name ${SOURCES}/main.cpp)









