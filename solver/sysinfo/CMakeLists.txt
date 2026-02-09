cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
project(sysinfo LANGUAGES CXX)

add_library(sysinfo STATIC sysinfo.h sysinfo.cpp)

target_include_directories(sysinfo PUBLIC .)