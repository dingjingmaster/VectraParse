cmake_minimum_required(VERSION 3.16)
project(date LANGUAGES CXX)

add_library(date::date INTERFACE IMPORTED GLOBAL)
set_target_properties(date::date PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/include")
