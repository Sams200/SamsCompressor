cmake_minimum_required(VERSION 3.29)
project(IPProject C)

set(CMAKE_C_STANDARD 11)

set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O0")

find_package(Threads REQUIRED)

add_executable(IPProject
        main.c
        bmp.c
        sams.c
        sams.h
        coder.c
        coder.h
)

target_link_libraries(IPProject m ${CMAKE_THREAD_LIBS_INIT})