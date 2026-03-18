#include "vector_store.h"
#include <sstream>
#include <iostream>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <mutex>

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
        if (db_) {
            sqlite3_close(db_);
        }
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
        int rc = sqlite3_enable_load_extension(db, 1);
        if (rc != SQLITE_OK) {
            error_msg = "Failed to enable extension loading";
            return false;
        }
        
        char* err = nullptr;
        
        // 尝试从指定路径加载
        if (!dll_path.empty()) {
            rc = sqlite3_load_extension(db, dll_path.c_str(), nullptr, &err);
            if (rc == SQLITE_OK) {
                return true;
            }
            if (err) sqlite3_free(err);
            err = nullptr;
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
            rc = sqlite3_load_extension(db, path, nullptr, &err);
            if (rc == SQLITE_OK) {
                return true;
            }
            if (err) {
                sqlite3_free(err);
                err = nullptr;
            }
        }
        
        error_msg = "Failed to load vec0.dll from any location";
        return false;
    }
};

bool VectorStore::Impl::LoadVecExtension(std::string& error_msg) {
    // 尝试多个可能的路径
    std::vector<std::string> try_paths = {
        // 相对于当前工作目录
        "third_party/vec0",
        "third_party/vec0.dll",
        "./third_party/vec0",
        "./third_party/vec0.dll",
        // 相对于可执行文件目录（需要在运行时确定）
    };
    
    // 先尝试简单路径
    for (const auto& path : try_paths) {
        if (SqliteVecExtension::Load(db_, path, error_msg)) {
            vec_extension_loaded_ = true;
            return true;
        }
    }
    
    vec_extension_loaded_ = false;
    return false;
}

VectorStore::VectorStore(const VectorStoreConfig& config)
    : impl_(std::make_unique<Impl>()), config_(config) {
}

VectorStore::~VectorStore() = default;
VectorStore::VectorStore(VectorStore&&) noexcept = default;
VectorStore& VectorStore::operator=(VectorStore&&) noexcept = default;

bool VectorStore::Initialize() {
    // 打开数据库
    int rc = sqlite3_open(config_.db_path.c_str(), &impl_->db_);
    if (rc != SQLITE_OK) {
        last_error_ = "Failed to open database: " + std::string(sqlite3_errmsg(impl_->db_));
        return false;
    }
    
    // 加载 vec0 扩展
    std::string ext_error;
    if (!impl_->LoadVecExtension(ext_error)) {
        std::cerr << "[Warning] Failed to load sqlite-vec extension: " << ext_error << std::endl;
        std::cerr << "[Warning] Falling back to pure SQL implementation" << std::endl;
        impl_->vec_extension_loaded_ = false;
    } else {
        std::cout << "[SQLite-Vec] Extension loaded successfully" << std::endl;
        impl_->vec_extension_loaded_ = true;
    }
    
    // 创建单表（使用 BLOB 存储向量，不使用虚拟表）
    std::stringstream sql;
    sql << "CREATE TABLE IF NOT EXISTS " << config_.table_name << " ("
        << "rowid INTEGER PRIMARY KEY, "
        << "content TEXT, "
        << "embedding BLOB, "  // 存储 float 数组
        << "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);";
    
    if (!impl_->ExecuteSQL(sql.str())) {
        last_error_ = "Failed to create table: " + std::string(sqlite3_errmsg(impl_->db_));
        return false;
    }
    
    // 创建索引加速查询
    std::string index_sql = "CREATE INDEX IF NOT EXISTS idx_" + config_.table_name + "_rowid ON " 
                          + config_.table_name + "(rowid);";
    impl_->ExecuteSQL(index_sql);
    
    is_initialized_ = true;
    std::cout << "[SQLite3 Backend] VectorStore initialized: " << config_.db_path 
              << " (Extension: " << (impl_->vec_extension_loaded_ ? "YES" : "NO") << ")" << std::endl;
    return true;
}

int64_t VectorStore::InsertVector(int64_t row_id, const Vector& vector, const std::string& content) {
    if (!is_initialized_) {
        last_error_ = "VectorStore not initialized";
        return -1;
    }
    
    if ((int)vector.size() != config_.vector_dimension) {
        last_error_ = "Vector dimension mismatch: expected " + 
                      std::to_string(config_.vector_dimension) + 
                      ", got " + std::to_string(vector.size());
        return -1;
    }
    
    // 如果 row_id 为 -1，获取下一个可用的 ID
    if (row_id < 0) {
        sqlite3_stmt* stmt;
        std::string sql = "SELECT COALESCE(MAX(rowid), 0) + 1 FROM " + config_.table_name + ";";
        int rc = sqlite3_prepare_v2(impl_->db_, sql.c_str(), -1, &stmt, nullptr);
        if (rc == SQLITE_OK && sqlite3_step(stmt) == SQLITE_ROW) {
            row_id = sqlite3_column_int64(stmt, 0);
        }
        sqlite3_finalize(stmt);
    }
    
    // 单表插入
    sqlite3_stmt* stmt;
    std::string sql = "INSERT OR REPLACE INTO " + config_.table_name + 
                      " (rowid, content, embedding) VALUES (?, ?, ?);";
    
    int rc = sqlite3_prepare_v2(impl_->db_, sql.c_str(), -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        last_error_ = "Failed to prepare insert: " + std::string(sqlite3_errmsg(impl_->db_));
        return -1;
    }
    
    sqlite3_bind_int64(stmt, 1, row_id);
    sqlite3_bind_text(stmt, 2, content.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_blob(stmt, 3, vector.data(), 
                      vector.size() * sizeof(float), SQLITE_STATIC);
    
    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    
    if (rc != SQLITE_DONE) {
        last_error_ = "Failed to insert: " + std::string(sqlite3_errmsg(impl_->db_));
        return -1;
    }
    
    return row_id;
}

bool VectorStore::InsertVectors(const std::vector<std::pair<Vector, std::string>>& vectors) {
    if (!is_initialized_) return false;
    
    // 开始事务
    impl_->ExecuteSQL("BEGIN TRANSACTION;");
    
    bool success = true;
    for (const auto& [vec, content] : vectors) {
        if (InsertVector(-1, vec, content) < 0) {
            success = false;
            break;
        }
    }
    
    if (success) {
        impl_->ExecuteSQL("COMMIT;");
    } else {
        impl_->ExecuteSQL("ROLLBACK;");
    }
    
    return success;
}

std::vector<SearchResult> VectorStore::SearchSimilar(const Vector& query_vector, int top_k) {
    std::vector<SearchResult> results;
    
    if (!is_initialized_) {
        last_error_ = "VectorStore not initialized";
        return results;
    }
    
    if ((int)query_vector.size() != config_.vector_dimension) {
        last_error_ = "Query vector dimension mismatch";
        return results;
    }
    
    // 如果加载了 sqlite-vec 扩展，尝试使用它的距离函数
    if (impl_->vec_extension_loaded_) {
        // 尝试使用 vec_distance_cosine 函数
        sqlite3_stmt* test_stmt;
        std::string test_sql = "SELECT vec_distance_cosine(?, ?);";
        if (sqlite3_prepare_v2(impl_->db_, test_sql.c_str(), -1, &test_stmt, nullptr) == SQLITE_OK) {
            sqlite3_finalize(test_stmt);
            
            // 使用 sqlite-vec 的距离函数
            sqlite3_stmt* stmt;
            std::string sql = 
                "SELECT rowid, content, vec_distance_cosine(embedding, ?) as distance "
                "FROM " + config_.table_name + " "
                "WHERE embedding IS NOT NULL "
                "ORDER BY distance "
                "LIMIT ?;";
            
            int rc = sqlite3_prepare_v2(impl_->db_, sql.c_str(), -1, &stmt, nullptr);
            if (rc == SQLITE_OK) {
                sqlite3_bind_blob(stmt, 1, query_vector.data(), 
                                 query_vector.size() * sizeof(float), SQLITE_STATIC);
                sqlite3_bind_int(stmt, 2, top_k);
                
                while (sqlite3_step(stmt) == SQLITE_ROW) {
                    int64_t rowid = sqlite3_column_int64(stmt, 0);
                    const char* content = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
                    double distance = sqlite3_column_double(stmt, 2);
                    results.emplace_back(rowid, static_cast<float>(distance), 
                                        content ? content : "");
                }
                sqlite3_finalize(stmt);
                return results;
            }
        }
    }
    
    // 回退：手动计算距离（纯 SQL 实现）
    std::vector<std::pair<int64_t, float>> distances;
    std::unordered_map<int64_t, std::string> content_map;
    
    sqlite3_stmt* stmt;
    std::string sql = "SELECT rowid, content, embedding FROM " + config_.table_name + 
                      " WHERE embedding IS NOT NULL;";
    
    int rc = sqlite3_prepare_v2(impl_->db_, sql.c_str(), -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        last_error_ = "Failed to prepare search";
        return results;
    }
    
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
    }
    sqlite3_finalize(stmt);
    
    // 排序并取前 k 个
    std::sort(distances.begin(), distances.end(), 
              [](const auto& a, const auto& b) { return a.second < b.second; });
    
    int count = std::min(top_k, (int)distances.size());
    for (int i = 0; i < count; ++i) {
        int64_t rowid = distances[i].first;
        std::string content;
        auto it = content_map.find(rowid);
        if (it != content_map.end()) {
            content = it->second;
        }
        results.emplace_back(rowid, distances[i].second, content);
    }
    
    return results;
}

std::vector<SearchResult> VectorStore::SearchSimilarWithFilter(
    const Vector& query_vector, int top_k, const std::string& where_clause) {
    // 简化实现
    return SearchSimilar(query_vector, top_k);
}

bool VectorStore::DeleteVector(int64_t row_id) {
    if (!is_initialized_) return false;
    
    std::string sql = "DELETE FROM " + config_.table_name + " WHERE rowid = ?;";
    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(impl_->db_, sql.c_str(), -1, &stmt, nullptr);
    if (rc != SQLITE_OK) return false;
    
    sqlite3_bind_int64(stmt, 1, row_id);
    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    
    return rc == SQLITE_DONE;
}

bool VectorStore::UpdateVector(int64_t row_id, const Vector& vector, const std::string& content) {
    return InsertVector(row_id, vector, content) >= 0;
}

int64_t VectorStore::GetVectorCount() {
    if (!is_initialized_) return -1;
    
    std::string sql = "SELECT COUNT(*) FROM " + config_.table_name + " WHERE embedding IS NOT NULL;";
    sqlite3_stmt* stmt;
    int64_t count = 0;
    
    int rc = sqlite3_prepare_v2(impl_->db_, sql.c_str(), -1, &stmt, nullptr);
    if (rc == SQLITE_OK && sqlite3_step(stmt) == SQLITE_ROW) {
        count = sqlite3_column_int64(stmt, 0);
    }
    sqlite3_finalize(stmt);
    
    return count;
}

bool VectorStore::ClearAll() {
    if (!is_initialized_) return false;
    
    std::string sql = "DELETE FROM " + config_.table_name + ";";
    return impl_->ExecuteSQL(sql);
}

void* VectorStore::GetDbHandle() const {
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
    impl_->memory_store = std::make_unique<MemoryVectorStore>();
}

VectorStore::~VectorStore() = default;
VectorStore::VectorStore(VectorStore&&) noexcept = default;
VectorStore& VectorStore::operator=(VectorStore&&) noexcept = default;

bool VectorStore::Initialize() {
    is_initialized_ = true;
    std::cout << "[Memory Backend] VectorStore initialized (no SQLite3 dependency)" << std::endl;
    return true;
}

int64_t VectorStore::InsertVector(int64_t row_id, const Vector& vector, const std::string& content) {
    if (!is_initialized_) {
        last_error_ = "VectorStore not initialized";
        return -1;
    }
    
    if ((int)vector.size() != config_.vector_dimension) {
        last_error_ = "Vector dimension mismatch";
        return -1;
    }
    
    std::lock_guard<std::mutex> lock(impl_->memory_store->mutex);
    
    if (row_id < 0) {
        row_id = impl_->memory_store->next_id++;
    } else {
        impl_->memory_store->next_id = std::max(impl_->memory_store->next_id, row_id + 1);
    }
    
    impl_->memory_store->data[row_id] = {vector, content};
    return row_id;
}

bool VectorStore::InsertVectors(const std::vector<std::pair<Vector, std::string>>& vectors) {
    for (const auto& [vec, content] : vectors) {
        if (InsertVector(-1, vec, content) < 0) {
            return false;
        }
    }
    return true;
}

std::vector<SearchResult> VectorStore::SearchSimilar(const Vector& query_vector, int top_k) {
    std::vector<SearchResult> results;
    
    if (!is_initialized_) {
        last_error_ = "VectorStore not initialized";
        return results;
    }
    
    if ((int)query_vector.size() != config_.vector_dimension) {
        last_error_ = "Query vector dimension mismatch";
        return results;
    }
    
    std::lock_guard<std::mutex> lock(impl_->memory_store->mutex);
    
    std::vector<std::pair<int64_t, float>> distances;
    
    for (const auto& [row_id, data] : impl_->memory_store->data) {
        const auto& [vec, _] = data;
        float dist = CalculateCosineDistance(query_vector.data(), vec.data(), config_.vector_dimension);
        distances.emplace_back(row_id, dist);
    }
    
    std::sort(distances.begin(), distances.end(), 
              [](const auto& a, const auto& b) { return a.second < b.second; });
    
    int count = std::min(top_k, (int)distances.size());
    for (int i = 0; i < count; ++i) {
        const auto& [vec, content] = impl_->memory_store->data[distances[i].first];
        results.emplace_back(distances[i].first, distances[i].second, content);
    }
    
    return results;
}

std::vector<SearchResult> VectorStore::SearchSimilarWithFilter(
    const Vector& query_vector, int top_k, const std::string& where_clause) {
    return SearchSimilar(query_vector, top_k);
}

bool VectorStore::DeleteVector(int64_t row_id) {
    std::lock_guard<std::mutex> lock(impl_->memory_store->mutex);
    return impl_->memory_store->data.erase(row_id) > 0;
}

bool VectorStore::UpdateVector(int64_t row_id, const Vector& vector, const std::string& content) {
    return InsertVector(row_id, vector, content) >= 0;
}

int64_t VectorStore::GetVectorCount() {
    std::lock_guard<std::mutex> lock(impl_->memory_store->mutex);
    return impl_->memory_store->data.size();
}

bool VectorStore::ClearAll() {
    std::lock_guard<std::mutex> lock(impl_->memory_store->mutex);
    impl_->memory_store->data.clear();
    return true;
}

void* VectorStore::GetDbHandle() const {
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
}

bool RAGQueryEngine::AddDocument(const std::string& text, const std::string& metadata) {
    auto vector = embedding_service_.Embed(text);
    return vector_store_.InsertVector(-1, vector, text) >= 0;
}

bool RAGQueryEngine::AddDocuments(const std::vector<std::string>& texts) {
    auto vectors = embedding_service_.EmbedBatch(texts);
    std::vector<std::pair<Vector, std::string>> data;
    for (size_t i = 0; i < texts.size(); ++i) {
        data.emplace_back(std::move(vectors[i]), texts[i]);
    }
    return vector_store_.InsertVectors(data);
}

std::vector<SearchResult> RAGQueryEngine::Query(const std::string& query_text, int top_k) {
    auto query_vector = embedding_service_.Embed(query_text);
    return vector_store_.SearchSimilar(query_vector, top_k);
}

} // namespace rag
