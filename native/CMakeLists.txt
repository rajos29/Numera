cmake_minimum_required(VERSION 3.16)

project(arithmetic_native_backend LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(arithmetic_benchmark src/main.cpp)

if(MSVC)
  target_compile_options(arithmetic_benchmark PRIVATE /O2)
else()
  target_compile_options(arithmetic_benchmark PRIVATE -O3)
endif()

