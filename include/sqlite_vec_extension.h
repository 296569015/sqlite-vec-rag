#pragma once

#include <string>

// 前置声明 sqlite3
typedef struct sqlite3 sqlite3;

namespace rag {

// sqlite-vec 扩展加载器
class SqliteVecExtension {
public:
    // 从指定路径加载 vec0.dll
    static bool Load(sqlite3* db, const std::string& dll_path, std::string& error_msg);
    
    // 检查扩展是否已加载
    static bool IsLoaded(sqlite3* db);
    
    // 获取版本
    static std::string GetVersion(sqlite3* db);
};

} // namespace rag
