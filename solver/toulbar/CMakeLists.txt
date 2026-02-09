set(PYTB2 OFF)
set(Boost OFF)
set(verbose OFF)
set(COVER_TEST OFF)
set(WITH_MEM_JEMALLOC OFF)
set(BUILD_SHARED_LIBS OFF)
set(STATIC ON)

cmake_minimum_required(VERSION 2.9)
IF(APPLE)
    cmake_minimum_required(VERSION 3.13)
ENDIF(APPLE)
cmake_policy(SET CMP0054 NEW)
project(toulbar2 LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set (My_cmake_script "cmake-script") # location of cmake script needed
set (My_Source src)	# source location
list(APPEND CMAKE_MODULE_PATH "${My_cmake_script}/Modules")

SET(CMAKE_MACOSX_RPATH 1)
include(CMakeDependentOption)

##########################################
# cmake option definition
##########################################
OPTION(LONG_COSTS  "use long long to encode costs [default:on]" ON)
OPTION(SHORT_COSTS  "use short int to encode costs [default:off]" OFF)
OPTION(BINARYWCSP  "do not store binary costs (only binary WCSP with no variable elimination during search) [default:off]" OFF)
OPTION(TERNARYWCSP  "do not store ternary costs (only ternary WCSP can be expressed) [default:off]" OFF)
OPTION(SHORT_VALUES  "use short int to encode domain values instead of int [default:off]" OFF)
OPTION(LONG_PROBABILITY  "use long double to encode probabilities [default:on]" ON)
OPTION(STATIC "static compilation flag [default: off]" ON)
OPTION(verbose "verbose mode [default:on]" OFF)

##########################################
#Default profile for compilation
##########################################
IF(APPLE) # Find homebrew libraries first
  message(STATUS "Apple/MacOS system detected. Selecting homebrew libraries first.")
  set(CMAKE_PREFIX_PATH /usr/local)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m64" )
  SET(STATIC OFF)
ENDIF(APPLE)

if(verbose) 
  set (CMAKE_VERBOSE_MAKEFILE ON)
endif(verbose)

##########################################
# static building
##########################################
IF(STATIC)
  MESSAGE(STATUS "STATIC COMPILATION ON")
  SET(BUILD_SHARED_LIBS OFF)
  set(CMAKE_SHARED_LIBRARY_LINK_CXX_FLAGS "")
  set(CMAKE_FIND_LIBRARY_SUFFIXES ".a")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static --disable-shared --enable-static -static-libgcc -static-libstdc++")
  SET (all_depends  ${all_depends} "z") 
  SET (all_depends  ${all_depends} "m") 
  SET (all_depends  ${all_depends} "pthread") 
  SET (all_depends  ${all_depends} "dl") # it should be present at the end of library ordering (checks manually no dynamic flags occur in the final link command)
ELSE(STATIC)
  SET(BUILD_SHARED_LIBS ON) #shared library building
  SET(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wall")
ENDIF(STATIC)

include(${My_cmake_script}/source_files.cmake)
SET(all_depends ${all_depends})

INCLUDE_DIRECTORIES (${CMAKE_CURRENT_SOURCE_DIR}/${My_Source} )

ADD_LIBRARY(tb2-objects OBJECT ${LIBTB2FILES})
ADD_LIBRARY(tb2-archive STATIC $<TARGET_OBJECTS:tb2-objects>)





