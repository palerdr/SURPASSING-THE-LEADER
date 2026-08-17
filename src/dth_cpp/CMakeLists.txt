cmake_minimum_required(VERSION 3.28)

project(dth_cpp LANGUAGES CXX)

include(CTest)

# This project uses C++20 language features but does not use C++ modules.
# Avoid CMake's default module-dependency scan and the associated compiler
# mapper flags in compile_commands.json.
set(CMAKE_CXX_SCAN_FOR_MODULES 0)

# Read the preset variable even while the requested scaffold has no compile
# targets. Once BUILD.md adds targets, CMake also uses it to emit the database.
if(CMAKE_EXPORT_COMPILE_COMMANDS)
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
endif()

option(DTH_ENABLE_SANITIZERS "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)

add_library(dth_project_options INTERFACE)
target_compile_features(dth_project_options INTERFACE cxx_std_20)

if(CMAKE_CXX_COMPILER_ID MATCHES "AppleClang|Clang|GNU")
    target_compile_options(
        dth_project_options
        INTERFACE
            -Wall
            -Wextra
            -Wpedantic
            -Wconversion
            -Wshadow
            -ffp-contract=off
    )
endif()

if(DTH_ENABLE_SANITIZERS AND CMAKE_CXX_COMPILER_ID MATCHES "AppleClang|Clang|GNU")
    target_compile_options(dth_project_options INTERFACE -fsanitize=address,undefined)
    target_link_options(dth_project_options INTERFACE -fsanitize=address,undefined)
endif()

# BUILD.md adds the library, executable, and test targets after their minimal
# translation units have been constructed. Keeping the scaffold target-free
# lets both presets configure while the requested implementation files are
# still empty.

find_package(Threads REQUIRED)

add_library(
    dth_solver
    STATIC
        storage/durable_store.cpp
        exact.cpp
        matrix_game.cpp
)

target_include_directories(
    dth_solver
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}"
)

if(WIN32)
    target_sources(dth_solver PRIVATE storage/mapped_file_win32.cpp)
else()
    target_sources(dth_solver PRIVATE storage/mapped_file_posix.cpp)
endif()

target_link_libraries(
    dth_solver
    PUBLIC
        Threads::Threads
    PRIVATE
        dth_project_options
)

add_executable(
    dth-solve-tablebase
    solve_tablebase.cpp
)

target_link_libraries(
    dth-solve-tablebase
    PRIVATE
        dth_solver
        dth_project_options
)

add_executable(
    dth-tests
    tests.cpp
)

target_link_libraries(
    dth-tests
    PRIVATE
        dth_solver
        dth_project_options
)

add_test(
    NAME dth-tests
    COMMAND dth-tests
)
