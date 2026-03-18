#include <string>
#include <iostream>
#include <windows.h>

#ifdef USE_SQLITE3
#include <sqlite3.h>

namespace rag {

class SqliteVecExtension {
public:
    // 从指定路径加载 vec0.dll
    static bool Load(sqlite3* db, const std::string& dll_path, std::string& error_msg) {
        // 首先启用扩展加载
        int rc = sqlite3_enable_load_extension(db, 1);
        if (rc != SQLITE_OK) {
            error_msg = "Failed to enable extension loading: " + std::string(sqlite3_errmsg(db));
            return false;
        }
        
        // 尝试从指定路径加载 vec0.dll
        char* err = nullptr;
        rc = sqlite3_load_extension(db, dll_path.c_str(), "sqlite3_vec_init", &err);
        
        if (rc != SQLITE_OK) {
            // 尝试只传入文件名（在 PATH 中查找）
            rc = sqlite3_load_extension(db, "vec0", nullptr, &err);
        }
        
        if (rc != SQLITE_OK) {
            error_msg = err ? err : "Failed to load vec0 extension";
            if (err) sqlite3_free(err);
            return false;
        }
        
        std::cout << "[SQLite-Vec] Extension loaded successfully from: " << dll_path << std::endl;
        return true;
    }
    
    // 检查扩展是否已加载
    static bool IsLoaded(sqlite3* db) {
        sqlite3_stmt* stmt;
        int rc = sqlite3_prepare_v2(db, "SELECT vec_version();", -1, &stmt, nullptr);
        if (rc == SQLITE_OK) {
            sqlite3_finalize(stmt);
            return true;
        }
        return false;
    }
    
    // 获取版本
    static std::string GetVersion(sqlite3* db) {
        sqlite3_stmt* stmt;
        if (sqlite3_prepare_v2(db, "SELECT vec_version();", -1, &stmt, nullptr) == SQLITE_OK) {
            if (sqlite3_step(stmt) == SQLITE_ROW) {
                const char* ver = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
                std::string result = ver ? ver : "unknown";
                sqlite3_finalize(stmt);
                return result;
            }
            sqlite3_finalize(stmt);
        }
        return "not loaded";
    }
};

} // namespace rag

#else

// Memory backend - no sqlite3 needed
namespace rag {

class SqliteVecExtension {
public:
    static bool Load(void* db, const std::string& dll_path, std::string& error_msg) { 
        return true; 
    }
    static bool IsLoaded(void* db) { 
        return true; 
    }
    static std::string GetVersion(void* db) { 
        return "memory-backend"; 
    }
};

} // namespace rag

#endif
