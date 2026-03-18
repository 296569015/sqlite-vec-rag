cmake_minimum_required(VERSION 3.16)
project(VectorSearch VERSION 1.0.0 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 选项
option(BUILD_EXAMPLES "Build examples" ON)

# SQLite3 配置 - 使用项目目录下的 sqlite3
set(SQLITE3_DIR "${CMAKE_CURRENT_SOURCE_DIR}/sqlite3")

if(NOT EXISTS "${SQLITE3_DIR}/sqlite3.c" OR NOT EXISTS "${SQLITE3_DIR}/sqlite3.h")
    message(FATAL_ERROR 
        "SQLite3 source files not found!\n"
        "Expected at: ${SQLITE3_DIR}\n"
        "Please ensure sqlite3.c and sqlite3.h exist in the sqlite3/ directory")
endif()

message(STATUS "Using SQLite3 from: ${SQLITE3_DIR}")

# Windows 下设置 UTF-8 编译选项
if(WIN32)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /utf-8")
endif()

# 创建 SQLite3 库
add_library(sqlite3 STATIC ${SQLITE3_DIR}/sqlite3.c)
target_include_directories(sqlite3 PUBLIC ${SQLITE3_DIR})
target_compile_definitions(sqlite3 PRIVATE
    SQLITE_ENABLE_COLUMN_METADATA=1
    SQLITE_ENABLE_RTREE=1
)

# 创建向量搜索库
add_library(vector_search STATIC
    src/vector_store.cpp
    src/sqlite_vec_extension.cpp
)

target_include_directories(vector_search
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${SQLITE3_DIR}
)

target_link_libraries(vector_search
    PUBLIC
        sqlite3
)

if(WIN32)
    target_link_libraries(vector_search PUBLIC ws2_32)
endif()

# 示例程序
if(BUILD_EXAMPLES)
    add_executable(vector_search_example examples/main.cpp)
    target_link_libraries(vector_search_example PRIVATE vector_search)
    set_target_properties(vector_search_example PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
endif()

# 打印配置信息
message(STATUS "============================================")
message(STATUS "VectorSearch Configuration:")
message(STATUS "  SQLite3: ${SQLITE3_DIR}")
message(STATUS "  Build examples: ${BUILD_EXAMPLES}")
message(STATUS "============================================")
