#include "vector_store.h"
#include "sqlite_vec_extension.h"
#include <sstream>
#include <iostream>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <unordered_map>
#include <unordered_set>
#include <sqlite3.h>

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

namespace rag {

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
    
    // 加载 vec0 扩展 - 必须成功，否则返回 false
    bool LoadVecExtension(std::string& error_msg) {
        LOG_ENTER();
        // 尝试多个可能的路径
        std::vector<std::string> try_paths = {
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
};

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
    
    // 加载 vec0 扩展 - 必须成功，这是 sqlite-vec 模式！
    LOG_INFO("Loading sqlite-vec extension (REQUIRED)...");
    std::string ext_error;
    if (!impl_->LoadVecExtension(ext_error)) {
        last_error_ = "Failed to load sqlite-vec extension: " + ext_error + 
                      ". This library requires vec0.dll to function. "
                      "Please download it from https://github.com/asg017/sqlite-vec/releases "
                      "and place it in the third_party/ directory.";
        LOG_ERROR(last_error_);
        sqlite3_close(impl_->db_);
        impl_->db_ = nullptr;
        LOG_EXIT();
        return false;
    }
    
    LOG_INFO("Extension loaded successfully");
    impl_->vec_extension_loaded_ = true;
    
    // 创建 vec0 虚拟表（不是普通表！）
    // 所有用于查询过滤的字段都必须加上 METADATA 关键字
    std::stringstream sql;
    sql << "CREATE VIRTUAL TABLE IF NOT EXISTS " << config_.table_name 
        << " USING vec0("
        << "embedding float[" << config_.vector_dimension << "], "  // 向量列
        // 业务元数据字段 (全部标记为 METADATA)
        << "convention_id TEXT METADATA, "       // 会话ID
        << "servermessage_id TEXT METADATA, "    // 服务器消息唯一ID
        << "recordtype TEXT METADATA, "          // 消息类型
        << "orinaccout TEXT METADATA, "          // 发送人账号
        << "msgTimestamp INTEGER METADATA, "     // 消息时间戳
        // 原有字段
        << "content TEXT METADATA, "             // 消息内容
        << "created_at TIMESTAMP METADATA"       // 入库时间
        << ");";
    
    LOG_INFO("Creating virtual table: " + config_.table_name);
    LOG_INFO("SQL: " + sql.str());
    
    if (!impl_->ExecuteSQL(sql.str())) {
        last_error_ = "Failed to create virtual table: " + std::string(sqlite3_errmsg(impl_->db_));
        LOG_ERROR(last_error_);
        sqlite3_close(impl_->db_);
        impl_->db_ = nullptr;
        LOG_EXIT();
        return false;
    }
    LOG_INFO("Virtual table created/verified successfully");
    
    // 注意：vec0 虚拟表自动管理索引，不需要手动创建
    
    is_initialized_ = true;
    LOG_INFO("VectorStore (sqlite-vec mode) initialized successfully");
    LOG_EXIT();
    return true;
}

int64_t VectorStore::InsertVector(int64_t row_id, const Vector& vector, const VectorMetadata& metadata) {
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
    
    LOG_INFO("Inserting vector with metadata: row_id=" + std::to_string(row_id) + 
             ", convention_id=" + metadata.convention_id + 
             ", recordtype=" + metadata.recordtype);
    
    // 构建 INSERT SQL，包含所有元数据字段
    std::string sql = "INSERT INTO " + config_.table_name + 
                      " (embedding, convention_id, servermessage_id, recordtype, "
                      "orinaccout, msgTimestamp, content, created_at) "
                      "VALUES (?, ?, ?, ?, ?, ?, ?, ?);";
    
    if (row_id >= 0) {
        // 如果指定了 rowid，使用 REPLACE
        sql = "INSERT OR REPLACE INTO " + config_.table_name + 
              " (rowid, embedding, convention_id, servermessage_id, recordtype, "
              "orinaccout, msgTimestamp, content, created_at) "
              "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);";
    }
    
    LOG_INFO("Preparing SQL: " + sql);
    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(impl_->db_, sql.c_str(), -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        last_error_ = "Failed to prepare insert: " + std::string(sqlite3_errmsg(impl_->db_));
        LOG_ERROR(last_error_);
        LOG_EXIT();
        return -1;
    }
    
    int bind_idx = 1;
    
    // 如果指定了 rowid，先绑定 rowid
    if (row_id >= 0) {
        sqlite3_bind_int64(stmt, bind_idx++, row_id);
    }
    
    // 绑定向量
    sqlite3_bind_blob(stmt, bind_idx++, vector.data(), 
                      vector.size() * sizeof(float), SQLITE_STATIC);
    
    // 绑定元数据字段
    sqlite3_bind_text(stmt, bind_idx++, metadata.convention_id.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt, bind_idx++, metadata.servermessage_id.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt, bind_idx++, metadata.recordtype.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt, bind_idx++, metadata.orinaccout.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_int64(stmt, bind_idx++, metadata.msgTimestamp);
    sqlite3_bind_text(stmt, bind_idx++, metadata.content.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt, bind_idx++, metadata.created_at.c_str(), -1, SQLITE_STATIC);
    
    LOG_INFO("Executing insert with all metadata...");
    rc = sqlite3_step(stmt);
    
    // 获取实际插入的 rowid
    if (row_id < 0) {
        row_id = sqlite3_last_insert_rowid(impl_->db_);
    }
    
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

bool VectorStore::InsertVectors(const std::vector<std::pair<Vector, VectorMetadata>>& vectors) {
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
    for (const auto& [vec, metadata] : vectors) {
        if (InsertVector(-1, vec, metadata) < 0) {
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
    
    // 使用 sqlite-vec 的 MATCH 语法进行向量搜索
    // 这是 vec0 虚拟表的核心功能！
    sqlite3_stmt* stmt;
    std::string sql = 
        "SELECT rowid, distance "
        "FROM " + config_.table_name + " "
        "WHERE embedding MATCH ? "
        "ORDER BY distance "
        "LIMIT ?;";
    
    LOG_INFO("Preparing search SQL: " + sql);
    int rc = sqlite3_prepare_v2(impl_->db_, sql.c_str(), -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        last_error_ = "Failed to prepare search: " + std::string(sqlite3_errmsg(impl_->db_));
        LOG_ERROR(last_error_);
        LOG_EXIT();
        return results;
    }
    
    // 绑定向量作为查询
    sqlite3_bind_blob(stmt, 1, query_vector.data(), 
                     query_vector.size() * sizeof(float), SQLITE_STATIC);
    sqlite3_bind_int(stmt, 2, top_k);
    
    LOG_INFO("Executing search...");
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        int64_t rowid = sqlite3_column_int64(stmt, 0);
        double distance = sqlite3_column_double(stmt, 1);
        results.emplace_back(rowid, static_cast<float>(distance), "");
        LOG_INFO("Found: rowid=" + std::to_string(rowid) + ", distance=" + std::to_string(distance));
    }
    
    sqlite3_finalize(stmt);
    LOG_INFO("Search complete: " + std::to_string(results.size()) + " results");
    LOG_EXIT();
    return results;
}

std::vector<SearchResult> VectorStore::SearchSimilarWithFilter(
    const Vector& query_vector, int top_k, const std::unordered_map<std::string, std::string>& filters) {
    LOG_ENTER();
    LOG_INFO("Filter search: filters_count=" + std::to_string(filters.size()));
    
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
    
    if (filters.empty()) {
        LOG_INFO("No filters, using regular search");
        LOG_EXIT();
        return SearchSimilar(query_vector, top_k);
    }
    
    // 构建安全的参数化查询
    // 只允许特定的元数据字段，防止 SQL 注入
    static const std::unordered_set<std::string> allowed_fields = {
        "convention_id", "servermessage_id", "recordtype", 
        "orinaccout", "msgTimestamp", "content", "created_at"
    };
    
    std::vector<std::string> valid_filters;
    std::vector<std::string> filter_values;
    
    for (const auto& [key, value] : filters) {
        if (allowed_fields.find(key) != allowed_fields.end()) {
            valid_filters.push_back(key);
            filter_values.push_back(value);
        } else {
            LOG_ERROR("Invalid filter field: " + key);
        }
    }
    
    if (valid_filters.empty()) {
        LOG_ERROR("No valid filter fields, using regular search");
        LOG_EXIT();
        return SearchSimilar(query_vector, top_k);
    }
    
    // 构建 SQL（字段名是硬编码的，值用参数绑定）
    std::string sql = "SELECT rowid, distance FROM " + config_.table_name + 
                      " WHERE embedding MATCH ?";
    
    for (const auto& field : valid_filters) {
        sql += " AND " + field + " = ?";
    }
    
    sql += " ORDER BY distance LIMIT ?;";
    
    LOG_INFO("Preparing filter search SQL: " + sql);
    
    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(impl_->db_, sql.c_str(), -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        last_error_ = "Failed to prepare filter search: " + std::string(sqlite3_errmsg(impl_->db_));
        LOG_ERROR(last_error_);
        LOG_EXIT();
        return results;
    }
    
    // 绑定参数
    int bind_idx = 1;
    
    // 绑定向量
    sqlite3_bind_blob(stmt, bind_idx++, query_vector.data(), 
                     query_vector.size() * sizeof(float), SQLITE_STATIC);
    
    // 绑定过滤值（所有值作为 TEXT）
    for (const auto& value : filter_values) {
        sqlite3_bind_text(stmt, bind_idx++, value.c_str(), -1, SQLITE_STATIC);
    }
    
    // 绑定 limit
    sqlite3_bind_int(stmt, bind_idx++, top_k);
    
    LOG_INFO("Executing filter search...");
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        int64_t rowid = sqlite3_column_int64(stmt, 0);
        double distance = sqlite3_column_double(stmt, 1);
        results.emplace_back(rowid, static_cast<float>(distance), "");
        LOG_INFO("Found: rowid=" + std::to_string(rowid) + ", distance=" + std::to_string(distance));
    }
    
    sqlite3_finalize(stmt);
    LOG_INFO("Filter search complete: " + std::to_string(results.size()) + " results");
    LOG_EXIT();
    return results;
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

bool VectorStore::UpdateVector(int64_t row_id, const Vector& vector, const VectorMetadata& metadata) {
    LOG_ENTER();
    LOG_INFO("Updating row_id=" + std::to_string(row_id));
    LOG_EXIT();
    return InsertVector(row_id, vector, metadata) >= 0;
}

int64_t VectorStore::GetVectorCount() {
    LOG_ENTER();
    
    if (!is_initialized_) {
        LOG_ERROR("VectorStore not initialized");
        LOG_EXIT();
        return -1;
    }
    
    std::string sql = "SELECT COUNT(*) FROM " + config_.table_name + ";";
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

// 静态函数：计算余弦距离
static float CalculateCosineDistance(const float* a, const float* b, int dim) {
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

bool RAGQueryEngine::AddDocument(const std::string& text, const std::string& doc_metadata) {
    LOG_ENTER();
    LOG_INFO("Adding document: " + text.substr(0, 50) + "...");
    auto vector = embedding_service_.Embed(text);
    LOG_INFO("Embedding generated, dim=" + std::to_string(vector.size()));
    
    // 构建元数据
    VectorMetadata meta;
    meta.content = text;
    meta.created_at = "2024-01-01 00:00:00"; // TODO: 使用实际时间
    // doc_metadata 可以解析为其他字段
    
    bool success = vector_store_.InsertVector(-1, vector, meta) >= 0;
    LOG_INFO("Insert " + std::string(success ? "successful" : "failed"));
    LOG_EXIT();
    return success;
}

bool RAGQueryEngine::AddDocuments(const std::vector<std::string>& texts) {
    LOG_ENTER();
    LOG_INFO("Adding " + std::to_string(texts.size()) + " documents");
    auto vectors = embedding_service_.EmbedBatch(texts);
    LOG_INFO("Generated " + std::to_string(vectors.size()) + " embeddings");
    std::vector<std::pair<Vector, VectorMetadata>> data;
    for (size_t i = 0; i < texts.size(); ++i) {
        VectorMetadata meta;
        meta.content = texts[i];
        meta.created_at = "2024-01-01 00:00:00"; // TODO: 使用实际时间
        data.emplace_back(std::move(vectors[i]), std::move(meta));
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
