#include "vector_store.h"
#include <sstream>
#include <iostream>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <mutex>
#include <chrono>
#include <iomanip>

// 日志宏定义
#define LOG_ENTER() \
    std::cout << "[LOG][ENTER] " << __FUNCTION__ << "() at " << GetCurrentTime() << std::endl

#define LOG_EXIT() \
    std::cout << "[LOG][EXIT] " << __FUNCTION__ << "() at " << GetCurrentTime() << std::endl

#define LOG_INFO(msg) \
    std::cout << "[LOG][INFO] " << __FUNCTION__ << "(): " << msg << std::endl

#define LOG_ERROR(msg) \
    std::cerr << "[LOG][ERROR] " << __FUNCTION__ << "(): " << msg << std::endl

// 获取当前时间字符串
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

// 根据宏定义选择后端
#ifdef USE_SQLITE3
    #include <sqlite3.h>
    #include <filesystem>
    #define USE_REAL_SQLITE
#else
    // 内存后端
#endif

namespace rag {

// ==================== SQLite3 + sqlite-vec 后端实现 ====================
#ifdef USE_REAL_SQLITE

class SqliteVecExtension;

class VectorStore::Impl {
public:
    sqlite3* db_ = nullptr;
    bool vec_extension_loaded_ = false;
    
    ~Impl() {
        LOG_ENTER();
        if (db_) {
            std::cout << "[LOG] Closing SQLite3 database" << std::endl;
            sqlite3_close(db_);
        }
        LOG_EXIT();
    }
    
    bool ExecuteSQL(const std::string& sql) {
        char* err_msg = nullptr;
        int rc = sqlite3_exec(db_, sql.c_str(), nullptr, nullptr, &err_msg);
        if (rc != SQLITE_OK && err_msg) {
            sqlite3_free(err_msg);
        }
        return rc == SQLITE_OK;
    }
    
    // 加载 vec0 扩展
    bool LoadVecExtension(std::string& error_msg);
};

// 前置声明
class SqliteVecExtension {
public:
    static bool Load(sqlite3* db, const std::string& dll_path, std::string& error_msg) {
        LOG_ENTER();
        std::cout << "[LOG] Attempting to load vec0.dll from: " << dll_path << std::endl;
        
        int rc = sqlite3_enable_load_extension(db, 1);
        if (rc != SQLITE_OK) {
            error_msg = "Failed to enable extension loading";
            LOG_ERROR(error_msg);
            LOG_EXIT();
            return false;
        }
        LOG_INFO("Extension loading enabled");
        
        char* err = nullptr;
        
        // 尝试从指定路径加载
        if (!dll_path.empty()) {
            std::cout << "[LOG] Trying path: " << dll_path << std::endl;
            rc = sqlite3_load_extension(db, dll_path.c_str(), nullptr, &err);
            if (rc == SQLITE_OK) {
                LOG_INFO("SUCCESS - loaded from: " + dll_path);
                LOG_EXIT();
                return true;
            }
            if (err) {
                std::cout << "[LOG] Failed: " << err << std::endl;
                sqlite3_free(err);
                err = nullptr;
            }
        }
        
        // 尝试其他常见路径
        const char* paths[] = {
            "./vec0",
            "./third_party/vec0",
            "vec0",
            "third_party/vec0",
            "./vec0.dll",
            "./third_party/vec0.dll"
        };
        
        for (const auto* path : paths) {
            std::cout << "[LOG] Trying path: " << path << std::endl;
            rc = sqlite3_load_extension(db, path, nullptr, &err);
            if (rc == SQLITE_OK) {
                LOG_INFO("SUCCESS - loaded from: " + std::string(path));
                LOG_EXIT();
                return true;
            }
            if (err) {
                std::cout << "[LOG] Failed: " << err << std::endl;
                sqlite3_free(err);
                err = nullptr;
            }
        }
        
        error_msg = "Failed to load vec0.dll from any location";
        LOG_ERROR(error_msg);
        LOG_EXIT();
        return false;
    }
};

bool VectorStore::Impl::LoadVecExtension(std::string& error_msg) {
    LOG_ENTER();
    // 尝试多个可能的路径
    std::vector<std::string> try_paths = {
        // 相对于当前工作目录
        "third_party/vec0",
        "third_party/vec0.dll",
        "./third_party/vec0",
        "./third_party/vec0.dll",
    };
    
    // 先尝试简单路径
    for (const auto& path : try_paths) {
        if (SqliteVecExtension::Load(db_, path, error_msg)) {
            vec_extension_loaded_ = true;
            LOG_EXIT();
            return true;
        }
    }
    
    vec_extension_loaded_ = false;
    LOG_EXIT();
    return false;
}

VectorStore::VectorStore(const VectorStoreConfig& config)
    : impl_(std::make_unique<Impl>()), config_(config) {
    LOG_ENTER();
    LOG_INFO("Config: db_path=" + config.db_path + ", table=" + config.table_name + ", dim=" + std::to_string(config.vector_dimension));
    LOG_EXIT();
}

VectorStore::~VectorStore() {
    LOG_ENTER();
    LOG_EXIT();
}

VectorStore::VectorStore(VectorStore&&) noexcept = default;
VectorStore& VectorStore::operator=(VectorStore&&) noexcept = default;

bool VectorStore::Initialize() {
    LOG_ENTER();
    
    // 打开数据库
    LOG_INFO("Opening database: " + config_.db_path);
    int rc = sqlite3_open(config_.db_path.c_str(), &impl_->db_);
    if (rc != SQLITE_OK) {
        last_error_ = "Failed to open database: " + std::string(sqlite3_errmsg(impl_->db_));
        LOG_ERROR(last_error_);
        LOG_EXIT();
        return false;
    }
    LOG_INFO("Database opened successfully");
    
    // 加载 vec0 扩展
    LOG_INFO("Loading sqlite-vec extension...");
    std::string ext_error;
    if (!impl_->LoadVecExtension(ext_error)) {
        LOG_ERROR("Failed to load extension: " + ext_error);
        LOG_INFO("Falling back to pure SQL implementation");
        impl_->vec_extension_loaded_ = false;
    } else {
        LOG_INFO("Extension loaded successfully");
        impl_->vec_extension_loaded_ = true;
    }
    
    // 创建单表
    std::stringstream sql;
    sql << "CREATE TABLE IF NOT EXISTS " << config_.table_name << " ("
        << "rowid INTEGER PRIMARY KEY, "
        << "content TEXT, "
        << "embedding BLOB, "
        << "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);";
    
    LOG_INFO("Creating table: " + config_.table_name);
    if (!impl_->ExecuteSQL(sql.str())) {
        last_error_ = "Failed to create table: " + std::string(sqlite3_errmsg(impl_->db_));
        LOG_ERROR(last_error_);
        LOG_EXIT();
        return false;
    }
    LOG_INFO("Table created/verified");
    
    // 创建索引
    std::string index_sql = "CREATE INDEX IF NOT EXISTS idx_" + config_.table_name + "_rowid ON " 
                          + config_.table_name + "(rowid);";
    impl_->ExecuteSQL(index_sql);
    LOG_INFO("Index created/verified");
    
    is_initialized_ = true;
    LOG_INFO("VectorStore initialized successfully (Extension: " + std::string(impl_->vec_extension_loaded_ ? "YES" : "NO") + ")");
    LOG_EXIT();
    return true;
}

int64_t VectorStore::InsertVector(int64_t row_id, const Vector& vector, const std::string& content) {
    LOG_ENTER();
    
    if (!is_initialized_) {
        last_error_ = "VectorStore not initialized";
        LOG_ERROR(last_error_);
        LOG_EXIT();
        return -1;
    }
    
    if ((int)vector.size() != config_.vector_dimension) {
        last_error_ = "Vector dimension mismatch: expected " + 
                      std::to_string(config_.vector_dimension) + 
                      ", got " + std::to_string(vector.size());
        LOG_ERROR(last_error_);
        LOG_EXIT();
        return -1;
    }
    
    LOG_INFO("Inserting vector: row_id=" + std::to_string(row_id) + ", content_len=" + std::to_string(content.length()));
    
    // 如果 row_id 为 -1，获取下一个可用的 ID
    if (row_id < 0) {
        LOG_INFO("Auto-generating row_id...");
        sqlite3_stmt* stmt;
        std::string sql = "SELECT COALESCE(MAX(rowid), 0) + 1 FROM " + config_.table_name + ";";
        int rc = sqlite3_prepare_v2(impl_->db_, sql.c_str(), -1, &stmt, nullptr);
        if (rc == SQLITE_OK && sqlite3_step(stmt) == SQLITE_ROW) {
            row_id = sqlite3_column_int64(stmt, 0);
            LOG_INFO("Generated row_id: " + std::to_string(row_id));
        }
        sqlite3_finalize(stmt);
    }
    
    // 单表插入
    sqlite3_stmt* stmt;
    std::string sql = "INSERT OR REPLACE INTO " + config_.table_name + 
                      " (rowid, content, embedding) VALUES (?, ?, ?);";
    
    LOG_INFO("Preparing SQL: " + sql);
    int rc = sqlite3_prepare_v2(impl_->db_, sql.c_str(), -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        last_error_ = "Failed to prepare insert: " + std::string(sqlite3_errmsg(impl_->db_));
        LOG_ERROR(last_error_);
        LOG_EXIT();
        return -1;
    }
    
    sqlite3_bind_int64(stmt, 1, row_id);
    sqlite3_bind_text(stmt, 2, content.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_blob(stmt, 3, vector.data(), 
                      vector.size() * sizeof(float), SQLITE_STATIC);
    
    LOG_INFO("Executing insert...");
    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    
    if (rc != SQLITE_DONE) {
        last_error_ = "Failed to insert: " + std::string(sqlite3_errmsg(impl_->db_));
        LOG_ERROR(last_error_);
        LOG_EXIT();
        return -1;
    }
    
    LOG_INFO("Insert successful: row_id=" + std::to_string(row_id));
    LOG_EXIT();
    return row_id;
}

bool VectorStore::InsertVectors(const std::vector<std::pair<Vector, std::string>>& vectors) {
    LOG_ENTER();
    LOG_INFO("Batch insert: " + std::to_string(vectors.size()) + " vectors");
    
    if (!is_initialized_) {
        LOG_ERROR("VectorStore not initialized");
        LOG_EXIT();
        return false;
    }
    
    // 开始事务
    LOG_INFO("Beginning transaction...");
    impl_->ExecuteSQL("BEGIN TRANSACTION;");
    
    bool success = true;
    int count = 0;
    for (const auto& [vec, content] : vectors) {
        if (InsertVector(-1, vec, content) < 0) {
            success = false;
            LOG_ERROR("Failed at index " + std::to_string(count));
            break;
        }
        count++;
        if (count % 10 == 0) {
            LOG_INFO("Inserted " + std::to_string(count) + "/" + std::to_string(vectors.size()));
        }
    }
    
    if (success) {
        LOG_INFO("Committing transaction...");
        impl_->ExecuteSQL("COMMIT;");
        LOG_INFO("Transaction committed");
    } else {
        LOG_ERROR("Rolling back transaction...");
        impl_->ExecuteSQL("ROLLBACK;");
        LOG_INFO("Transaction rolled back");
    }
    
    LOG_EXIT();
    return success;
}

std::vector<SearchResult> VectorStore::SearchSimilar(const Vector& query_vector, int top_k) {
    LOG_ENTER();
    LOG_INFO("Search: top_k=" + std::to_string(top_k) + ", query_dim=" + std::to_string(query_vector.size()));
    
    std::vector<SearchResult> results;
    
    if (!is_initialized_) {
        last_error_ = "VectorStore not initialized";
        LOG_ERROR(last_error_);
        LOG_EXIT();
        return results;
    }
    
    if ((int)query_vector.size() != config_.vector_dimension) {
        last_error_ = "Query vector dimension mismatch";
        LOG_ERROR(last_error_);
        LOG_EXIT();
        return results;
    }
    
    // 如果加载了 sqlite-vec 扩展，尝试使用它的距离函数
    if (impl_->vec_extension_loaded_) {
        LOG_INFO("Trying to use vec_distance_cosine()...");
        sqlite3_stmt* test_stmt;
        std::string test_sql = "SELECT vec_distance_cosine(?, ?);";
        if (sqlite3_prepare_v2(impl_->db_, test_sql.c_str(), -1, &test_stmt, nullptr) == SQLITE_OK) {
            sqlite3_finalize(test_stmt);
            LOG_INFO("vec_distance_cosine() is available");
            
            sqlite3_stmt* stmt;
            std::string sql = 
                "SELECT rowid, content, vec_distance_cosine(embedding, ?) as distance "
                "FROM " + config_.table_name + " "
                "WHERE embedding IS NOT NULL "
                "ORDER BY distance "
                "LIMIT ?;";
            
            LOG_INFO("Preparing search SQL with extension...");
            int rc = sqlite3_prepare_v2(impl_->db_, sql.c_str(), -1, &stmt, nullptr);
            if (rc == SQLITE_OK) {
                sqlite3_bind_blob(stmt, 1, query_vector.data(), 
                                 query_vector.size() * sizeof(float), SQLITE_STATIC);
                sqlite3_bind_int(stmt, 2, top_k);
                
                LOG_INFO("Executing search...");
                while (sqlite3_step(stmt) == SQLITE_ROW) {
                    int64_t rowid = sqlite3_column_int64(stmt, 0);
                    const char* content = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
                    double distance = sqlite3_column_double(stmt, 2);
                    results.emplace_back(rowid, static_cast<float>(distance), 
                                        content ? content : "");
                    LOG_INFO("Found: rowid=" + std::to_string(rowid) + ", distance=" + std::to_string(distance));
                }
                sqlite3_finalize(stmt);
                LOG_INFO("Search complete: " + std::to_string(results.size()) + " results");
                LOG_EXIT();
                return results;
            }
            LOG_ERROR("Failed to prepare search SQL with extension");
        } else {
            LOG_INFO("vec_distance_cosine() not available, falling back");
        }
    } else {
        LOG_INFO("Extension not loaded, using pure SQL implementation");
    }
    
    // 回退：手动计算距离
    LOG_INFO("Using fallback: manual distance calculation");
    std::vector<std::pair<int64_t, float>> distances;
    std::unordered_map<int64_t, std::string> content_map;
    
    sqlite3_stmt* stmt;
    std::string sql = "SELECT rowid, content, embedding FROM " + config_.table_name + 
                      " WHERE embedding IS NOT NULL;";
    
    LOG_INFO("Querying all vectors...");
    int rc = sqlite3_prepare_v2(impl_->db_, sql.c_str(), -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        last_error_ = "Failed to prepare search";
        LOG_ERROR(last_error_);
        LOG_EXIT();
        return results;
    }
    
    int scanned = 0;
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        int64_t rowid = sqlite3_column_int64(stmt, 0);
        const char* content = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
        const void* blob = sqlite3_column_blob(stmt, 2);
        int blob_size = sqlite3_column_bytes(stmt, 2);
        
        if (blob && blob_size == config_.vector_dimension * sizeof(float)) {
            const float* vec_data = static_cast<const float*>(blob);
            float dist = CalculateCosineDistance(query_vector.data(), 
                                                  vec_data, 
                                                  config_.vector_dimension);
            distances.emplace_back(rowid, dist);
            if (content) content_map[rowid] = content;
        }
        scanned++;
    }
    sqlite3_finalize(stmt);
    LOG_INFO("Scanned " + std::to_string(scanned) + " vectors");
    
    // 排序并取前 k 个
    std::sort(distances.begin(), distances.end(), 
              [](const auto& a, const auto& b) { return a.second < b.second; });
    
    int count = std::min(top_k, (int)distances.size());
    LOG_INFO("Returning top " + std::to_string(count) + " results");
    
    for (int i = 0; i < count; ++i) {
        int64_t rowid = distances[i].first;
        std::string content;
        auto it = content_map.find(rowid);
        if (it != content_map.end()) {
            content = it->second;
        }
        results.emplace_back(rowid, distances[i].second, content);
        LOG_INFO("Result #" + std::to_string(i+1) + ": rowid=" + std::to_string(rowid) + ", dist=" + std::to_string(distances[i].second));
    }
    
    LOG_EXIT();
    return results;
}

std::vector<SearchResult> VectorStore::SearchSimilarWithFilter(
    const Vector& query_vector, int top_k, const std::string& where_clause) {
    LOG_ENTER();
    LOG_INFO("Filter search not fully implemented, using regular search");
    LOG_EXIT();
    return SearchSimilar(query_vector, top_k);
}

bool VectorStore::DeleteVector(int64_t row_id) {
    LOG_ENTER();
    LOG_INFO("Deleting row_id=" + std::to_string(row_id));
    
    if (!is_initialized_) {
        LOG_ERROR("VectorStore not initialized");
        LOG_EXIT();
        return false;
    }
    
    std::string sql = "DELETE FROM " + config_.table_name + " WHERE rowid = ?;";
    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(impl_->db_, sql.c_str(), -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        LOG_ERROR("Failed to prepare delete");
        LOG_EXIT();
        return false;
    }
    
    sqlite3_bind_int64(stmt, 1, row_id);
    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    
    bool success = (rc == SQLITE_DONE);
    LOG_INFO("Delete " + std::string(success ? "successful" : "failed"));
    LOG_EXIT();
    return success;
}

bool VectorStore::UpdateVector(int64_t row_id, const Vector& vector, const std::string& content) {
    LOG_ENTER();
    LOG_INFO("Updating row_id=" + std::to_string(row_id));
    LOG_EXIT();
    return InsertVector(row_id, vector, content) >= 0;
}

int64_t VectorStore::GetVectorCount() {
    LOG_ENTER();
    
    if (!is_initialized_) {
        LOG_ERROR("VectorStore not initialized");
        LOG_EXIT();
        return -1;
    }
    
    std::string sql = "SELECT COUNT(*) FROM " + config_.table_name + " WHERE embedding IS NOT NULL;";
    sqlite3_stmt* stmt;
    int64_t count = 0;
    
    int rc = sqlite3_prepare_v2(impl_->db_, sql.c_str(), -1, &stmt, nullptr);
    if (rc == SQLITE_OK && sqlite3_step(stmt) == SQLITE_ROW) {
        count = sqlite3_column_int64(stmt, 0);
    }
    sqlite3_finalize(stmt);
    
    LOG_INFO("Vector count: " + std::to_string(count));
    LOG_EXIT();
    return count;
}

bool VectorStore::ClearAll() {
    LOG_ENTER();
    
    if (!is_initialized_) {
        LOG_ERROR("VectorStore not initialized");
        LOG_EXIT();
        return false;
    }
    
    LOG_INFO("Clearing all data from table: " + config_.table_name);
    std::string sql = "DELETE FROM " + config_.table_name + ";";
    bool success = impl_->ExecuteSQL(sql);
    LOG_INFO("Clear " + std::string(success ? "successful" : "failed"));
    LOG_EXIT();
    return success;
}

void* VectorStore::GetDbHandle() const {
    LOG_ENTER();
    LOG_INFO("Returning db handle");
    LOG_EXIT();
    return impl_->db_;
}

// ==================== 内存后端实现 ====================
#else

struct MemoryVectorStore {
    std::unordered_map<int64_t, std::pair<Vector, std::string>> data;
    std::mutex mutex;
    int64_t next_id = 1;
};

class VectorStore::Impl {
public:
    std::unique_ptr<MemoryVectorStore> memory_store;
};

VectorStore::VectorStore(const VectorStoreConfig& config)
    : impl_(std::make_unique<Impl>()), config_(config) {
    LOG_ENTER();
    LOG_INFO("Memory backend - Config: table=" + config.table_name + ", dim=" + std::to_string(config.vector_dimension));
    impl_->memory_store = std::make_unique<MemoryVectorStore>();
    LOG_EXIT();
}

VectorStore::~VectorStore() {
    LOG_ENTER();
    LOG_EXIT();
}

VectorStore::VectorStore(VectorStore&&) noexcept = default;
VectorStore& VectorStore::operator=(VectorStore&&) noexcept = default;

bool VectorStore::Initialize() {
    LOG_ENTER();
    is_initialized_ = true;
    LOG_INFO("Memory backend initialized");
    LOG_EXIT();
    return true;
}

int64_t VectorStore::InsertVector(int64_t row_id, const Vector& vector, const std::string& content) {
    LOG_ENTER();
    
    if (!is_initialized_) {
        last_error_ = "VectorStore not initialized";
        LOG_ERROR(last_error_);
        LOG_EXIT();
        return -1;
    }
    
    if ((int)vector.size() != config_.vector_dimension) {
        last_error_ = "Vector dimension mismatch";
        LOG_ERROR(last_error_);
        LOG_EXIT();
        return -1;
    }
    
    std::lock_guard<std::mutex> lock(impl_->memory_store->mutex);
    
    if (row_id < 0) {
        row_id = impl_->memory_store->next_id++;
    } else {
        impl_->memory_store->next_id = std::max(impl_->memory_store->next_id, row_id + 1);
    }
    
    impl_->memory_store->data[row_id] = {vector, content};
    LOG_INFO("Inserted: row_id=" + std::to_string(row_id));
    LOG_EXIT();
    return row_id;
}

bool VectorStore::InsertVectors(const std::vector<std::pair<Vector, std::string>>& vectors) {
    LOG_ENTER();
    LOG_INFO("Batch insert: " + std::to_string(vectors.size()) + " vectors");
    
    for (const auto& [vec, content] : vectors) {
        if (InsertVector(-1, vec, content) < 0) {
            LOG_ERROR("Batch insert failed");
            LOG_EXIT();
            return false;
        }
    }
    
    LOG_INFO("Batch insert successful");
    LOG_EXIT();
    return true;
}

std::vector<SearchResult> VectorStore::SearchSimilar(const Vector& query_vector, int top_k) {
    LOG_ENTER();
    LOG_INFO("Search: top_k=" + std::to_string(top_k));
    
    std::vector<SearchResult> results;
    
    if (!is_initialized_) {
        last_error_ = "VectorStore not initialized";
        LOG_ERROR(last_error_);
        LOG_EXIT();
        return results;
    }
    
    if ((int)query_vector.size() != config_.vector_dimension) {
        last_error_ = "Query vector dimension mismatch";
        LOG_ERROR(last_error_);
        LOG_EXIT();
        return results;
    }
    
    std::lock_guard<std::mutex> lock(impl_->memory_store->mutex);
    
    LOG_INFO("Scanning " + std::to_string(impl_->memory_store->data.size()) + " vectors");
    std::vector<std::pair<int64_t, float>> distances;
    
    for (const auto& [row_id, data] : impl_->memory_store->data) {
        const auto& [vec, _] = data;
        float dist = CalculateCosineDistance(query_vector.data(), vec.data(), config_.vector_dimension);
        distances.emplace_back(row_id, dist);
    }
    
    std::sort(distances.begin(), distances.end(), 
              [](const auto& a, const auto& b) { return a.second < b.second; });
    
    int count = std::min(top_k, (int)distances.size());
    LOG_INFO("Returning top " + std::to_string(count) + " results");
    
    for (int i = 0; i < count; ++i) {
        const auto& [vec, content] = impl_->memory_store->data[distances[i].first];
        results.emplace_back(distances[i].first, distances[i].second, content);
    }
    
    LOG_EXIT();
    return results;
}

std::vector<SearchResult> VectorStore::SearchSimilarWithFilter(
    const Vector& query_vector, int top_k, const std::string& where_clause) {
    LOG_ENTER();
    LOG_EXIT();
    return SearchSimilar(query_vector, top_k);
}

bool VectorStore::DeleteVector(int64_t row_id) {
    LOG_ENTER();
    LOG_INFO("Deleting row_id=" + std::to_string(row_id));
    std::lock_guard<std::mutex> lock(impl_->memory_store->mutex);
    bool success = impl_->memory_store->data.erase(row_id) > 0;
    LOG_INFO("Delete " + std::string(success ? "successful" : "failed"));
    LOG_EXIT();
    return success;
}

bool VectorStore::UpdateVector(int64_t row_id, const Vector& vector, const std::string& content) {
    LOG_ENTER();
    LOG_EXIT();
    return InsertVector(row_id, vector, content) >= 0;
}

int64_t VectorStore::GetVectorCount() {
    LOG_ENTER();
    std::lock_guard<std::mutex> lock(impl_->memory_store->mutex);
    int64_t count = impl_->memory_store->data.size();
    LOG_INFO("Vector count: " + std::to_string(count));
    LOG_EXIT();
    return count;
}

bool VectorStore::ClearAll() {
    LOG_ENTER();
    std::lock_guard<std::mutex> lock(impl_->memory_store->mutex);
    impl_->memory_store->data.clear();
    LOG_INFO("All data cleared");
    LOG_EXIT();
    return true;
}

void* VectorStore::GetDbHandle() const {
    LOG_ENTER();
    LOG_INFO("Memory backend - no db handle");
    LOG_EXIT();
    return nullptr;
}

#endif

// ==================== 公共实现 ====================

float VectorStore::CalculateCosineDistance(const float* a, const float* b, int dim) {
    float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
    for (int i = 0; i < dim; ++i) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    float denom = std::sqrt(norm_a) * std::sqrt(norm_b);
    return denom > 0 ? 1.0f - (dot / denom) : 2.0f;
}

// RAGQueryEngine 实现
RAGQueryEngine::RAGQueryEngine(VectorStore& vector_store, EmbeddingService& embedding_service)
    : vector_store_(vector_store), embedding_service_(embedding_service) {
    LOG_ENTER();
    LOG_INFO("RAGQueryEngine created");
    LOG_EXIT();
}

bool RAGQueryEngine::AddDocument(const std::string& text, const std::string& metadata) {
    LOG_ENTER();
    LOG_INFO("Adding document: " + text.substr(0, 50) + "...");
    auto vector = embedding_service_.Embed(text);
    LOG_INFO("Embedding generated, dim=" + std::to_string(vector.size()));
    bool success = vector_store_.InsertVector(-1, vector, text) >= 0;
    LOG_INFO("Insert " + std::string(success ? "successful" : "failed"));
    LOG_EXIT();
    return success;
}

bool RAGQueryEngine::AddDocuments(const std::vector<std::string>& texts) {
    LOG_ENTER();
    LOG_INFO("Adding " + std::to_string(texts.size()) + " documents");
    auto vectors = embedding_service_.EmbedBatch(texts);
    LOG_INFO("Generated " + std::to_string(vectors.size()) + " embeddings");
    std::vector<std::pair<Vector, std::string>> data;
    for (size_t i = 0; i < texts.size(); ++i) {
        data.emplace_back(std::move(vectors[i]), texts[i]);
    }
    bool success = vector_store_.InsertVectors(data);
    LOG_EXIT();
    return success;
}

std::vector<SearchResult> RAGQueryEngine::Query(const std::string& query_text, int top_k) {
    LOG_ENTER();
    LOG_INFO("Query: \"" + query_text + "\", top_k=" + std::to_string(top_k));
    auto query_vector = embedding_service_.Embed(query_text);
    LOG_INFO("Query vector generated, dim=" + std::to_string(query_vector.size()));
    auto results = vector_store_.SearchSimilar(query_vector, top_k);
    LOG_INFO("Retrieved " + std::to_string(results.size()) + " results");
    LOG_EXIT();
    return results;
}

} // namespace rag
