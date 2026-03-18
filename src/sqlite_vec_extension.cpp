// sqlite_vec_extension.cpp - SQLite-Vec Extension Loader

#include "sqlite_vec_extension.h"
#include <iostream>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <sqlite3.h>

// 日志宏
#define LOG_ENTER() \
    std::cout << "[EXT][ENTER] " << __FUNCTION__ << "() at " << GetCurrentTime() << std::endl

#define LOG_EXIT() \
    std::cout << "[EXT][EXIT] " << __FUNCTION__ << "() at " << GetCurrentTime() << std::endl

#define LOG_INFO(msg) \
    std::cout << "[EXT][INFO] " << __FUNCTION__ << "(): " << msg << std::endl

#define LOG_ERROR(msg) \
    std::cerr << "[EXT][ERROR] " << __FUNCTION__ << "(): " << msg << std::endl

static std::string GetCurrentTime() {
    auto now = std::chrono::system_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
    auto timer = std::chrono::system_clock::to_time_t(now);
    std::tm bt{};
    localtime_s(&bt, &timer);
    std::ostringstream oss;
    oss << std::put_time(&bt, "%H:%M:%S") << '.' << std::setfill('0') << std::setw(3) << ms.count();
    return oss.str();
}

namespace rag {

bool SqliteVecExtension::Load(sqlite3* db, const std::string& dll_path, std::string& error_msg) {
    LOG_ENTER();
    
    // 首先启用扩展加载
    LOG_INFO("Enabling extension loading...");
    int rc = sqlite3_enable_load_extension(db, 1);
    if (rc != SQLITE_OK) {
        error_msg = "Failed to enable extension loading: " + std::string(sqlite3_errmsg(db));
        LOG_ERROR(error_msg);
        LOG_EXIT();
        return false;
    }
    LOG_INFO("Extension loading enabled successfully");
    
    char* err = nullptr;
    
    // 尝试从指定路径加载
    if (!dll_path.empty()) {
        LOG_INFO("Attempting to load from: " + dll_path);
        rc = sqlite3_load_extension(db, dll_path.c_str(), "sqlite3_vec_init", &err);
        
        if (rc == SQLITE_OK) {
            LOG_INFO("SUCCESS - Extension loaded from: " + dll_path);
            LOG_EXIT();
            return true;
        }
        
        if (err) {
            LOG_ERROR("Failed to load from " + dll_path + ": " + err);
            sqlite3_free(err);
            err = nullptr;
        }
    }
    
    // 尝试其他常见路径
    LOG_INFO("Trying alternative paths...");
    const char* paths[] = {
        "./vec0",
        "./third_party/vec0",
        "vec0",
        "third_party/vec0",
        "./vec0.dll",
        "./third_party/vec0.dll"
    };
    
    for (const auto* path : paths) {
        LOG_INFO("Trying: " + std::string(path));
        rc = sqlite3_load_extension(db, path, nullptr, &err);
        if (rc == SQLITE_OK) {
            LOG_INFO("SUCCESS - Extension loaded from: " + std::string(path));
            LOG_EXIT();
            return true;
        }
        if (err) {
            LOG_ERROR("Failed: " + std::string(err));
            sqlite3_free(err);
            err = nullptr;
        }
    }
    
    error_msg = "Failed to load vec0.dll from any location";
    LOG_ERROR(error_msg);
    LOG_EXIT();
    return false;
}

bool SqliteVecExtension::IsLoaded(sqlite3* db) {
    LOG_ENTER();
    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(db, "SELECT vec_version();", -1, &stmt, nullptr);
    if (rc == SQLITE_OK) {
        sqlite3_finalize(stmt);
        LOG_INFO("Extension is loaded");
        LOG_EXIT();
        return true;
    }
    LOG_INFO("Extension is NOT loaded");
    LOG_EXIT();
    return false;
}

std::string SqliteVecExtension::GetVersion(sqlite3* db) {
    LOG_ENTER();
    sqlite3_stmt* stmt;
    if (sqlite3_prepare_v2(db, "SELECT vec_version();", -1, &stmt, nullptr) == SQLITE_OK) {
        if (sqlite3_step(stmt) == SQLITE_ROW) {
            const char* ver = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
            std::string result = ver ? ver : "unknown";
            sqlite3_finalize(stmt);
            LOG_INFO("Version: " + result);
            LOG_EXIT();
            return result;
        }
        sqlite3_finalize(stmt);
    }
    LOG_INFO("Version: not loaded");
    LOG_EXIT();
    return "not loaded";
}

} // namespace rag
