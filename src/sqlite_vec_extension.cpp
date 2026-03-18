#include <string>
#include <iostream>

#ifdef USE_SQLITE3
#include <sqlite3.h>

namespace rag {

class SqliteVecExtension {
public:
    static bool Load(sqlite3* db, std::string& error_msg) {
        sqlite3_enable_load_extension(db, 1);
        char* err = nullptr;
        if (sqlite3_load_extension(db, "vec0", nullptr, &err) != SQLITE_OK) {
            error_msg = err ? err : "Failed to load extension";
            if (err) sqlite3_free(err);
            return false;
        }
        return true;
    }
    
    static bool IsLoaded(sqlite3* db) {
        sqlite3_stmt* stmt;
        return sqlite3_prepare_v2(db, "SELECT vec_version();", -1, &stmt, nullptr) == SQLITE_OK;
    }
    
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
    static bool Load(void* db, std::string& error_msg) { 
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
