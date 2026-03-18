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
    #define USE_REAL_SQLITE
#else
    // 内存后端
#endif

namespace rag {

// ==================== SQLite3 后端实现 ====================
#ifdef USE_REAL_SQLITE

class VectorStore::Impl {
public:
    sqlite3* db_ = nullptr;
    
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
};

VectorStore::VectorStore(const VectorStoreConfig& config)
    : impl_(std::make_unique<Impl>()), config_(config) {
}

VectorStore::~VectorStore() = default;
VectorStore::VectorStore(VectorStore&&) noexcept = default;
VectorStore& VectorStore::operator=(VectorStore&&) noexcept = default;

bool VectorStore::Initialize() {
    int rc = sqlite3_open(config_.db_path.c_str(), &impl_->db_);
    if (rc != SQLITE_OK) {
        last_error_ = "Failed to open database: " + std::string(sqlite3_errmsg(impl_->db_));
        return false;
    }
    
    // 启用外键支持
    impl_->ExecuteSQL("PRAGMA foreign_keys = ON;");
    
    // 创建内容表
    std::stringstream content_sql;
    content_sql << "CREATE TABLE IF NOT EXISTS " << config_.table_name << "_content ("
                << "rowid INTEGER PRIMARY KEY, "
                << "content TEXT, "
                << "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);";
    
    if (!impl_->ExecuteSQL(content_sql.str())) {
        last_error_ = "Failed to create content table";
        return false;
    }
    
    // 创建向量表
    std::stringstream vec_sql;
    vec_sql << "CREATE TABLE IF NOT EXISTS " << config_.table_name << "_vec ("
            << "rowid INTEGER PRIMARY KEY, "
            << "embedding BLOB);";
    
    if (!impl_->ExecuteSQL(vec_sql.str())) {
        last_error_ = "Failed to create vector table";
        return false;
    }
    
    is_initialized_ = true;
    std::cout << "[SQLite3 Backend] VectorStore initialized: " << config_.db_path << std::endl;
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
        std::string sql = "SELECT COALESCE(MAX(rowid), 0) + 1 FROM " + config_.table_name + "_content;";
        int rc = sqlite3_prepare_v2(impl_->db_, sql.c_str(), -1, &stmt, nullptr);
        if (rc == SQLITE_OK && sqlite3_step(stmt) == SQLITE_ROW) {
            row_id = sqlite3_column_int64(stmt, 0);
        }
        sqlite3_finalize(stmt);
    }
    
    // 插入内容
    sqlite3_stmt* content_stmt;
    std::string content_sql = "INSERT OR REPLACE INTO " + config_.table_name + 
                              "_content (rowid, content) VALUES (?, ?);";
    
    int rc = sqlite3_prepare_v2(impl_->db_, content_sql.c_str(), -1, &content_stmt, nullptr);
    if (rc != SQLITE_OK) {
        last_error_ = "Failed to prepare content insert";
        return -1;
    }
    
    sqlite3_bind_int64(content_stmt, 1, row_id);
    sqlite3_bind_text(content_stmt, 2, content.c_str(), -1, SQLITE_STATIC);
    
    rc = sqlite3_step(content_stmt);
    sqlite3_finalize(content_stmt);
    
    if (rc != SQLITE_DONE) {
        last_error_ = "Failed to insert content: " + std::string(sqlite3_errmsg(impl_->db_));
        return -1;
    }
    
    // 插入向量
    sqlite3_stmt* vec_stmt;
    std::string vec_sql = "INSERT OR REPLACE INTO " + config_.table_name + 
                          "_vec (rowid, embedding) VALUES (?, ?);";
    
    rc = sqlite3_prepare_v2(impl_->db_, vec_sql.c_str(), -1, &vec_stmt, nullptr);
    if (rc != SQLITE_OK) {
        last_error_ = "Failed to prepare vector insert";
        return -1;
    }
    
    sqlite3_bind_int64(vec_stmt, 1, row_id);
    sqlite3_bind_blob(vec_stmt, 2, vector.data(), 
                      vector.size() * sizeof(float), SQLITE_STATIC);
    
    rc = sqlite3_step(vec_stmt);
    sqlite3_finalize(vec_stmt);
    
    if (rc != SQLITE_DONE) {
        last_error_ = "Failed to insert vector: " + std::string(sqlite3_errmsg(impl_->db_));
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
    
    // 计算所有向量的距离
    std::vector<std::pair<int64_t, float>> distances;
    
    sqlite3_stmt* stmt;
    std::string sql = "SELECT rowid, embedding FROM " + config_.table_name + "_vec;";
    
    int rc = sqlite3_prepare_v2(impl_->db_, sql.c_str(), -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        last_error_ = "Failed to prepare search";
        return results;
    }
    
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        int64_t rowid = sqlite3_column_int64(stmt, 0);
        const void* blob = sqlite3_column_blob(stmt, 1);
        int blob_size = sqlite3_column_bytes(stmt, 1);
        
        if (blob && blob_size == config_.vector_dimension * sizeof(float)) {
            const float* vec_data = static_cast<const float*>(blob);
            float dist = CalculateCosineDistance(query_vector.data(), 
                                                  vec_data, 
                                                  config_.vector_dimension);
            distances.emplace_back(rowid, dist);
        }
    }
    sqlite3_finalize(stmt);
    
    // 排序并取前 k 个
    std::sort(distances.begin(), distances.end(), 
              [](const auto& a, const auto& b) { return a.second < b.second; });
    
    int count = std::min(top_k, (int)distances.size());
    for (int i = 0; i < count; ++i) {
        // 获取内容
        std::string content;
        std::string content_sql = "SELECT content FROM " + config_.table_name + 
                                  "_content WHERE rowid = ?;";
        sqlite3_stmt* content_stmt;
        rc = sqlite3_prepare_v2(impl_->db_, content_sql.c_str(), -1, &content_stmt, nullptr);
        if (rc == SQLITE_OK) {
            sqlite3_bind_int64(content_stmt, 1, distances[i].first);
            if (sqlite3_step(content_stmt) == SQLITE_ROW) {
                const char* text = reinterpret_cast<const char*>(sqlite3_column_text(content_stmt, 0));
                if (text) content = text;
            }
            sqlite3_finalize(content_stmt);
        }
        
        results.emplace_back(distances[i].first, distances[i].second, content);
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
    
    std::string sql = "DELETE FROM " + config_.table_name + "_vec WHERE rowid = ?;";
    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(impl_->db_, sql.c_str(), -1, &stmt, nullptr);
    if (rc != SQLITE_OK) return false;
    
    sqlite3_bind_int64(stmt, 1, row_id);
    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    
    if (rc != SQLITE_DONE) return false;
    
    // 同时删除内容
    sql = "DELETE FROM " + config_.table_name + "_content WHERE rowid = ?;";
    rc = sqlite3_prepare_v2(impl_->db_, sql.c_str(), -1, &stmt, nullptr);
    if (rc == SQLITE_OK) {
        sqlite3_bind_int64(stmt, 1, row_id);
        sqlite3_step(stmt);
        sqlite3_finalize(stmt);
    }
    
    return true;
}

bool VectorStore::UpdateVector(int64_t row_id, const Vector& vector, const std::string& content) {
    return InsertVector(row_id, vector, content) >= 0;
}

int64_t VectorStore::GetVectorCount() {
    if (!is_initialized_) return -1;
    
    std::string sql = "SELECT COUNT(*) FROM " + config_.table_name + "_vec;";
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
    
    std::string sql = "DELETE FROM " + config_.table_name + "_vec;";
    bool success = impl_->ExecuteSQL(sql);
    
    sql = "DELETE FROM " + config_.table_name + "_content;";
    success &= impl_->ExecuteSQL(sql);
    
    return success;
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
