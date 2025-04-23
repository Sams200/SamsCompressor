cmake_minimum_required(VERSION 3.29)
project(IPProject C)

set(CMAKE_C_STANDARD 11)

add_executable(IPProject
        main.c
        bmp.c
        sams.c
        sams.h
        jpeg.c
        jpeg.h
)

target_link_libraries(IPProject m)