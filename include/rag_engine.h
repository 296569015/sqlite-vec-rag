#pragma once

// RAG 引擎主入口头文件
// 包含所有必要的头文件，方便外部项目使用

#include "vector_store.h"

// 版本信息
#define RAG_VERSION_MAJOR 1
#define RAG_VERSION_MINOR 0
#define RAG_VERSION_PATCH 0

namespace rag {

// 版本信息
inline const char* GetVersion() {
    return "1.0.0";
}

} // namespace rag
