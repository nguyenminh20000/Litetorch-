cmake_minimum_required(VERSION 3.10)
project(litetorch LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(include)

file(GLOB_RECURSE SOURCES "src/*.cpp")

find_package(Threads REQUIRED)

add_library(litetorch SHARED ${SOURCES})
target_link_libraries(litetorch PRIVATE ${CMAKE_DL_LIBS} Threads::Threads)


file(GLOB TEST_SOURCES "tests/*.cpp")
foreach(test_source ${TEST_SOURCES})
    get_filename_component(test_name ${test_source} NAME_WE)
    add_executable(${test_name} ${test_source})
    target_link_libraries(${test_name} PRIVATE litetorch Threads::Threads)
endforeach()
