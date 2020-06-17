###
### CMake settings
###
cmake_minimum_required(VERSION 2.6)
cmake_policy(SET CMP0012 NEW)


###
### Project settings
###
project(NoRELAX)
set(NORELAX_VERSION_MAJOR "0")
set(NORELAX_VERSION_MINOR "1")
set(NORELAX_VERSION_PATCH "0")
set(NORELAX_VERSION "${NORELAX_VERSION_MAJOR}.${NORELAX_VERSION_MINOR}.${NORELAX_VERSION_PATCH}")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

OPTION(WITH_OPENGM "Include OpenGM" ON)

## For compatibility with older GCC
## add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)

###
### Sources, headers, directories and libs
###
include_directories(${CMAKE_SOURCE_DIR}/norelax)
include_directories(${CMAKE_SOURCE_DIR}/norelax/Eigen)

#find_package(GSL REQUIRED) 
#if(GSL_FOUND)
#	include_directories(${GSL_INCLUDE_DIRS})
#endif()    

set(SRC
    "norelax/main.cpp"
    norelax/HMRF.hpp
    norelax/HMRF.cpp
    norelax/PairwiseMRF.hpp
    norelax/PairwiseMRF.cpp
    norelax/DenseMRF.hpp
    norelax/DenseMRF.cpp
    norelax/PairwiseMRF.cpp
    norelax/lemmas.hpp
    norelax/lemmas.cpp
    norelax/outils_io.hpp
    norelax/OpenGM.hpp
    norelax/makeunique.hpp
    norelax/cxxopts.hpp
    norelax/projection.h
)

if (WITH_OPENGM)
    message(STATUS "build with opengm")
    SET(OPENGM_INCLUDE_DIR "" CACHE STRING "Include directory for OpenGM")
    SET(OPENGM_LIB CACHE STRING "Library for OpenGM")
    include_directories(${OPENGM_INCLUDE_DIR})
    add_definitions(-DWITH_OPENGM)

    message(STATUS "build with HDF5 support")
    if(WIN32)
      # FindHDF5 is broken on Windows
      # it wrongly discovers the dlls instead of the lib files
      # see: www.cmake.org/Bug/bug_relationship_graph.php?bug_id=14111
      # therefore we set the variable manually
      find_library( HDF5_IMPORT_LIB NAMES hdf5 hdf5dll )
      find_library( HDF5_HL_IMPORT_LIB NAMES hdf5_hl hdf5_hldll )
      find_path(HDF5_INCLUDE_DIRS hdf5.h)
      set( HDF5_LIBRARIES ${HDF5_IMPORT_LIB} ${HDF5_HL_IMPORT_LIB} )
      message(STATUS "HDF5: autodiscovery is broken on WIN32; using global search paths")
      message(STATUS "HDF5: found ${HDF5_LIBRARIES}")
    else()
      find_package(HDF5 REQUIRED)       
    endif()
    include_directories(${HDF5_INCLUDE_DIRS})
    add_definitions(${HDF5_CPPFLAGS})
    add_definitions(-DWITH_HDF5)
    LINK_DIRECTORIES(${HDF5_LIBRARY_DIRS})
    include_directories(${HDF5_LIBRARY_DIRS})
    set(_hdf5_libs hdf5 hdf5_cpp)
else(WITH_OPENGM)
    message(STATUS "build without OpenGM support")
endif(WITH_OPENGM)


  


###
### Build configuration
### 
set(CMAKE_BUILD_TYPE Release)

# GCC settings
#
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_MINSIZEREL "-Os")
#
set(GCC_WARNING_FLAGS "-Wall -Wextra -pedantic -Wno-missing-braces -Woverloaded-virtual -Wno-unused-parameter")
#
set(CMAKE_CXX_FLAGS "--std=c++11  -O3 -fopenmp ${GCC_WARNING_FLAGS} ${CMAKE_CXX_FLAGS}")

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

#

###
### Target: libnorelax
###

# add_library(NoRELAX STATIC ${SRC})


add_executable(NoRELAX ${SRC})

## target_link_libraries(NoRELAX ${GSL_LIBRARIES} ${_hdf5_libs})
target_link_libraries(NoRELAX ${_hdf5_libs})

