cmake_minimum_required(VERSION 3.28)

project(dth_cpp LANGUAGES CXX)

include(CTest)

include(FetchContent)

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
option(
    DTH_FETCH_HIGHS
    "Fetch the pinned HiGHS source when an exact installed package is unavailable"
    ON
)

# Numerical optimization is delegated to one pinned HiGHS release. Prefer an
# exact, toolchain-compatible installation; otherwise FetchContent builds the
# same release as part of this build.
find_package(HIGHS 1.15.1 EXACT CONFIG QUIET)

if(NOT TARGET highs::highs)
    if(NOT DTH_FETCH_HIGHS)
        message(
            FATAL_ERROR
            "HiGHS 1.15.1 was not found. Set HIGHS_DIR or enable DTH_FETCH_HIGHS."
        )
    endif()

    set(_dth_saved_build_testing "${BUILD_TESTING}")
    set(_dth_saved_build_shared_libs "${BUILD_SHARED_LIBS}")

    set(BUILD_TESTING OFF CACHE BOOL "Build tests" FORCE)
    set(BUILD_CXX ON CACHE BOOL "Build the HiGHS C++ library" FORCE)
    set(BUILD_CXX_EXE OFF CACHE BOOL "Build the HiGHS executable" FORCE)
    set(BUILD_EXAMPLES OFF CACHE BOOL "Build HiGHS examples" FORCE)
    set(BUILD_SHARED_EXTRAS_LIB OFF CACHE BOOL "Build HiGHS extras" FORCE)
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build shared libraries" FORCE)
    set(FAST_BUILD ON CACHE BOOL "Use the HiGHS fast build" FORCE)
    set(HIPO OFF CACHE BOOL "Build HiPO" FORCE)
    set(ZLIB OFF CACHE BOOL "Enable zlib in HiGHS" FORCE)
    set(HIGHS_NO_DEFAULT_THREADS ON CACHE BOOL "Disable implicit HiGHS threading" FORCE)

    FetchContent_Declare(
        highs
        GIT_REPOSITORY https://github.com/ERGO-Code/HiGHS.git
        GIT_TAG 04024d701f79feb8e2f18bc3df0dffc04ef05088
        GIT_PROGRESS TRUE
    )
    FetchContent_MakeAvailable(highs)

    set(BUILD_TESTING "${_dth_saved_build_testing}" CACHE BOOL "Build tests" FORCE)
    set(BUILD_SHARED_LIBS "${_dth_saved_build_shared_libs}" CACHE BOOL "Build shared libraries" FORCE)
    unset(_dth_saved_build_testing)
    unset(_dth_saved_build_shared_libs)
endif()

if(NOT TARGET highs::highs)
    message(FATAL_ERROR "The pinned HiGHS build did not provide highs::highs")
endif()

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

# Keep the solver in one library so the CLI and both independent gate
# executables exercise the same implementation.

find_package(Threads REQUIRED)

add_library(
    dth_solver
    STATIC
        storage/durable_store.cpp
        exact.cpp
        highs_backend.cpp
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
        highs::highs
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

add_executable(
    dth-highs-backend-tests
    highs_backend_tests.cpp
)

target_link_libraries(
    dth-highs-backend-tests
    PRIVATE
        dth_solver
        dth_project_options
)

add_test(
    NAME dth-highs-backend-tests
    COMMAND dth-highs-backend-tests
)
