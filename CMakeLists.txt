cmake_minimum_required(VERSION 3.20)
project(
    Weft
    VERSION 1.0
    LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE
        Release
        CACHE STRING "Build type" FORCE)
endif()
include(CheckIPOSupported)
check_ipo_supported(RESULT WEFT_IPO_SUPPORTED)
if(WEFT_IPO_SUPPORTED)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
endif()
find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED)
find_package(nlohmann_json 3.11 QUIET)
include(FetchContent)
if(NOT nlohmann_json_FOUND)
    if(POLICY CMP0135)
        cmake_policy(SET CMP0135 NEW)
    endif()
    set(JSON_BuildTests
        OFF
        CACHE BOOL "" FORCE)
    FetchContent_Declare(
        json
        URL https://codeload.github.com/nlohmann/json/tar.gz/refs/tags/v3.11.3
        URL_HASH SHA256=0d8ef5af7f9794e3263480193c491549b2ba6cc74bb018906202ada498a79406)
    FetchContent_MakeAvailable(json)
endif()
add_library(
    bj
    src/prg.cpp
    src/config.cpp
    src/net.cpp
    src/open_buf.cpp
    src/material.cpp
    src/dealer_guard.cpp
    src/input.cpp
    src/party.cpp
    src/compare_shape.cpp
    src/comparison_dealer.cpp
    src/masked_material.cpp
    src/investigation_plan.cpp
    src/shuffle.cpp
    src/bandpair.cpp
    src/compact.cpp
    src/nl.cpp
    src/dc_plan.cpp
    src/dc.cpp
    src/pattern_plan.cpp
    src/pattern_nl.cpp
    src/pattern_dc.cpp
    src/query_post.cpp
    src/multi_plan.cpp
    src/multi_query.cpp
    src/multi_round.cpp
    src/shared_sort.cpp
    src/multi_cost.cpp
    src/shared_interval.cpp
    src/bank_data.cpp
    src/plain_reference.cpp
    src/batch_suspect_plan.cpp
    src/batch_suspects.cpp)
target_include_directories(bj PUBLIC include)
target_compile_options(bj PUBLIC -march=native -maes -mpopcnt $<$<CONFIG:Release>:-O3>)
target_link_libraries(bj PUBLIC Threads::Threads OpenMP::OpenMP_CXX nlohmann_json::nlohmann_json)
add_executable(dealer_main dealer/dealer_main.cpp dealer/material.cpp dealer/comparison_dealer.cpp
                           dealer/masked_material.cpp)
target_link_libraries(dealer_main PRIVATE bj)
add_executable(party_main apps/party_main.cpp)
target_link_libraries(party_main PRIVATE bj)
foreach(name plain_reference plan_info)
    add_executable(${name} apps/${name}.cpp)
    target_link_libraries(${name} PRIVATE bj)
    set_target_properties(${name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/apps")
endforeach()
foreach(name band_bench compact_bench dc_bench)
    add_executable(${name} bench/investigation_bench.cpp tests/allocation.cpp)
    target_link_libraries(${name} PRIVATE bj)
    set_target_properties(${name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bench")
endforeach()
foreach(name prim_bench net_bench)
    add_executable(${name} bench/${name}.cpp)
    target_link_libraries(${name} PRIVATE bj)
    set_target_properties(${name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bench")
endforeach()
foreach(name investigate multi_bench)
    add_executable(${name} bench/multi_bench.cpp tests/allocation.cpp)
    target_link_libraries(${name} PRIVATE bj)
endforeach()
set_target_properties(investigate PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/apps")
set_target_properties(multi_bench PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bench")
add_executable(batch_suspects bench/batch_suspects_bench.cpp)
target_link_libraries(batch_suspects PRIVATE bj)
set_target_properties(batch_suspects PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                                "${CMAKE_BINARY_DIR}/bench")
option(WEFT_BUILD_TESTS "Build unit and parallel-kernel tests" OFF)
if(WEFT_BUILD_TESTS)
    enable_testing()
    find_package(GTest QUIET)
    if(NOT GTest_FOUND)
        set(BUILD_GMOCK
            OFF
            CACHE BOOL "" FORCE)
        set(INSTALL_GTEST
            OFF
            CACHE BOOL "" FORCE)
        FetchContent_Declare(
            googletest
            URL https://codeload.github.com/google/googletest/tar.gz/refs/tags/v1.15.2
            URL_HASH SHA256=7b42b4d6ed48810c5362c265a17faebe90dc2373c885e5216439d37927f02926)
        FetchContent_MakeAvailable(googletest)
    endif()
    add_executable(
        bj_tests
        tests/common_test.cpp
        tests/net_test.cpp
        tests/open_test.cpp
        tests/t1_test.cpp
        tests/t2_test.cpp
        tests/t45_test.cpp
        tests/test_dc_model.cpp
        tests/test_dc_pool.cpp
        tests/test_dc_mpc.cpp
        tests/test_v8.cpp
        tests/test_multi.cpp
        tests/test_t8.cpp
        tests/test_v9.cpp
        tests/allocation.cpp)
    target_link_libraries(bj_tests PRIVATE bj GTest::gtest_main)
    add_test(NAME unit COMMAND bj_tests)
    set_tests_properties(unit PROPERTIES TIMEOUT 180)
    add_executable(dealer_kernel_test tests/x11_parallel_dealer_test.cpp)
    target_link_libraries(dealer_kernel_test PRIVATE bj)
    add_test(NAME dealer_kernels COMMAND dealer_kernel_test)
    set_tests_properties(dealer_kernels PROPERTIES TIMEOUT 120)
endif()
