cmake_minimum_required(VERSION 3.14)
project(RATS VERSION 1.1 LANGUAGES CXX)


# Options: Things you can set via commandline options to cmake (e.g. -DENABLE_LTO=[ON|OFF])
option(ENABLE_WARNINGS_SETTINGS "Allow target_set_warnings to add flags and defines.
                                 Set this to OFF if you want to provide your own warning parameters." ON)
option(ENABLE_LTO "Enable link time optimization" ON)
option(ENABLE_DOCTESTS "Include tests in the library. Setting this to OFF will remove all doctest related code.
                        Tests in tests/*.cpp will still be enabled." OFF)
option(PEDANTIC "Enable pedantic compiler warnings." OFF)

# Include stuff.
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")
include(ConfigSafeGuards)
include(Colors)
include(CTest)
# include(Doctest)
include(Documentation)
include(LTO)
include(Misc)
include(Warnings)

# Check for LTO support.
find_lto(CXX)

# Find packages
if(NOT TARGET spdlog)
    find_package(spdlog REQUIRED)
endif()

if(NOT TARGET Eigen3)
    find_package(Eigen3 REQUIRED)
endif()

find_package(pybind11 REQUIRED)

find_package(ortools CONFIG REQUIRED)

# Define NDEBUG macro
add_definitions(-DNDEBUG)

# --------------------------------------------------------------------------------
#                         Locate files (change as needed).
# --------------------------------------------------------------------------------
set(SOURCES          # All .cpp files in src/
    src/example.cpp
    src/envs/investor_env.cpp
    src/rand.cpp
    src/utils.cpp
)
set(LIBRARY_NAME cpprats)  # Default name for the library built from src/*.cpp (change if you wish)

# --------------------------------------------------------------------------------
#                            Build library
# --------------------------------------------------------------------------------
# Compile all sources into a library.
add_library(${LIBRARY_NAME} OBJECT ${SOURCES})

# Lib needs its header files, and users of the library must also see these (PUBLIC). (No change needed)
target_include_directories(${LIBRARY_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/include)

# Link libraries
target_link_libraries(${LIBRARY_NAME} PRIVATE spdlog::spdlog $<$<BOOL:${MINGW}>:ws2_32>)
target_include_directories(${LIBRARY_NAME} SYSTEM PUBLIC ${EIGEN3_INCLUDE_DIR})
target_link_libraries(${LIBRARY_NAME} PUBLIC Eigen3::Eigen)
target_link_libraries(${LIBRARY_NAME} PRIVATE ortools::ortools)
target_link_libraries(${LIBRARY_NAME} PRIVATE pybind11::pybind11)

# Set the compile options you want (change as needed). Only if PEDANTIC is ON.
if(PEDANTIC)
    message(STATUS "Enabling pedantic compiler warnings for ${LIBRARY_NAME}. If this is annoying, set -DPEDANTIC=OFF.")
endif()

# --------------------------------------------------------------------------------
#                            Build targets
# --------------------------------------------------------------------------------

# Add the executeble
set(MAIN_TESTS_NAME regtests)
add_executable(${MAIN_TESTS_NAME} tests/main.cpp)

# Add the python binding
set(MAIN_MODULE_NAME _rats)
pybind11_add_module(${MAIN_MODULE_NAME} pybind/interface.cpp)

# foreach(_target ${MAIN_APP_NAME} ${MAIN_MODULE_NAME})
foreach(_target ${MAIN_MODULE_NAME} ${MAIN_TESTS_NAME})
# Link libraries
target_link_libraries(${_target} PRIVATE ${LIBRARY_NAME})  # Link the executable to library (if it uses it).
# target_link_libraries(${_target} PUBLIC doctest)
target_link_libraries(${_target} PRIVATE spdlog::spdlog $<$<BOOL:${MINGW}>:ws2_32>)
target_link_libraries(${_target} PUBLIC ortools::ortools)
target_link_libraries(${_target} PRIVATE pybind11::pybind11)

# Set the compilation options
if (PEDANTIC)
    message(STATUS "Enabling pedantic compiler warnings for ${_target}. If this is annoying, set -DPEDANTIC=OFF.")
endif()
target_enable_lto(${_target} optimized)  # enable link-time-optimization if available for non-debug configurations
endforeach()


# Set the properties you require, e.g. what C++ standard to use.
set_target_properties(
    ${LIBRARY_NAME} ${MAIN_TESTS_NAME}
      PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
        COMPILE_OPTIONS "-O2"
)

