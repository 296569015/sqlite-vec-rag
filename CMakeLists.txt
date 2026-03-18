cmake_minimum_required(VERSION 3.16)
project(VectorSearch VERSION 1.0.0 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 选项
option(BUILD_EXAMPLES "Build examples" ON)
option(USE_SQLITE3 "Use SQLite3 backend (requires sqlite3.h and sqlite3.c)" ON)

if(USE_SQLITE3)
    message(STATUS "Using SQLite3 backend...")
    
    # 查找 SQLite3 源文件（项目目录下的 sqlite3 文件夹）
    set(SQLITE3_DIR "${CMAKE_CURRENT_SOURCE_DIR}/sqlite3" CACHE PATH "SQLite3 source directory")
    
    if(NOT EXISTS "${SQLITE3_DIR}/sqlite3.c" OR NOT EXISTS "${SQLITE3_DIR}/sqlite3.h")
        message(FATAL_ERROR 
            "SQLite3 source files not found!\n"
            "Please download sqlite3.c and sqlite3.h from https://www.sqlite.org/download.html\n"
            "and place them in: ${SQLITE3_DIR}\n"
            "Or run: .\\download_sqlite3.ps1")
    endif()
    
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
    
    # Note: SQLITE_API defaults to empty for static library
    # No need to define __declspec(dllexport) for static builds
    
    set(SQLITE3_LIBRARY sqlite3)
    set(SQLITE3_INCLUDE_DIR ${SQLITE3_DIR})
    
    message(STATUS "SQLite3 source: ${SQLITE3_DIR}")
else()
    message(STATUS "Using in-memory backend (no SQLite3 dependency)")
endif()

# 创建向量搜索库
add_library(vector_search STATIC
    src/vector_store.cpp
    src/sqlite_vec_extension.cpp
)

target_include_directories(vector_search
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

if(USE_SQLITE3)
    target_include_directories(vector_search PRIVATE ${SQLITE3_INCLUDE_DIR})
    target_link_libraries(vector_search PUBLIC ${SQLITE3_LIBRARY})
    target_compile_definitions(vector_search PRIVATE USE_SQLITE3)
    
    if(WIN32)
        target_link_libraries(vector_search PUBLIC ws2_32)
    endif()
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
message(STATUS "  USE_SQLITE3: ${USE_SQLITE3}")
if(USE_SQLITE3)
    message(STATUS "  SQLITE3_INCLUDE_DIR: ${SQLITE3_INCLUDE_DIR}")
    message(STATUS "  SQLITE3_LIBRARY: ${SQLITE3_LIBRARY}")
else()
    message(STATUS "  Backend: In-Memory (no persistence)")
endif()
message(STATUS "============================================")
