#pragma once

#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <optional>
#include <unordered_map>

namespace rag {

// 向量类型定义
using Vector = std::vector<float>;

// 向量元数据结构 - 用于插入时携带业务字段
struct VectorMetadata {
    std::string convention_id;       // 会话ID
    std::string servermessage_id;    // 服务器消息唯一ID
    std::string recordtype;          // 消息类型 (如: "text", "image", "system")
    std::string orinaccout;          // 发送人账号
    int64_t msgTimestamp = 0;        // 消息时间戳 (Unix Timestamp)
    std::string content;             // 消息内容
    std::string created_at;          // 入库时间
};

// 搜索结果结构 - 包含完整元数据
struct SearchResult {
    int64_t row_id;
    float distance;
    VectorMetadata metadata;  // 完整元数据
    
    SearchResult(int64_t id = 0, float dist = 0.0f) 
        : row_id(id), distance(dist) {}
};

// 向量存储配置
struct VectorStoreConfig {
    std::string db_path = "vector_store.db";     // 数据库文件路径
    std::string table_name = "vectors";           // 表名
    int vector_dimension = 1024;                  // 向量维度
    std::string distance_metric = "cosine";       // 距离度量：cosine, l2, hamming
    bool use_index = true;                        // 是否使用索引
};

// 向量存储类 - 与 sqlite-vec 集成
class VectorStore {
public:
    explicit VectorStore(const VectorStoreConfig& config = {});
    ~VectorStore();
    
    // 禁止拷贝，允许移动
    VectorStore(const VectorStore&) = delete;
    VectorStore& operator=(const VectorStore&) = delete;
    VectorStore(VectorStore&&) noexcept;
    VectorStore& operator=(VectorStore&&) noexcept;
    
    // 初始化数据库和表
    bool Initialize();
    
    // 插入向量（带完整元数据）
    // @param row_id: 行ID（如果为-1则自动分配）
    // @param vector: 向量数据
    // @param metadata: 元数据（会话ID、消息类型、发送人等）
    // @return: 实际的 row_id，失败返回 -1
    int64_t InsertVector(int64_t row_id, const Vector& vector, const VectorMetadata& metadata);
    
    // 批量插入向量
    bool InsertVectors(const std::vector<std::pair<Vector, VectorMetadata>>& vectors);
    
    // 相似性搜索
    // @param query_vector: 查询向量
    // @param top_k: 返回最相似的 k 个结果
    // @return: 搜索结果列表
    std::vector<SearchResult> SearchSimilar(const Vector& query_vector, int top_k = 5);
    
    // 带过滤条件的搜索（安全的参数化查询）
    // @param query_vector: 查询向量
    // @param top_k: 返回最相似的 k 个结果
    // @param filters: 元数据过滤条件（key-value 对，支持 convention_id, recordtype, orinaccout 等）
    // @return: 搜索结果列表
    std::vector<SearchResult> SearchSimilarWithFilter(
        const Vector& query_vector, 
        int top_k, 
        const std::unordered_map<std::string, std::string>& filters
    );
    
    // 删除向量
    bool DeleteVector(int64_t row_id);
    
    // 更新向量
    bool UpdateVector(int64_t row_id, const Vector& vector, const VectorMetadata& metadata);
    
    // 获取向量数量
    int64_t GetVectorCount();
    
    // 清空所有数据
    bool ClearAll();
    
    // 获取最后错误信息
    std::string GetLastError() const { return last_error_; }
    
    // 检查是否就绪
    bool IsReady() const { return is_initialized_; }
    
    // 获取数据库句柄（用于高级操作）
    void* GetDbHandle() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
    VectorStoreConfig config_;
    std::string last_error_;
    bool is_initialized_ = false;
};

// 嵌入服务接口 - 与 native-emmbedding 项目集成
class EmbeddingService {
public:
    virtual ~EmbeddingService() = default;
    
    // 将文本转换为向量
    virtual Vector Embed(const std::string& text) = 0;
    
    // 批量嵌入
    virtual std::vector<Vector> EmbedBatch(const std::vector<std::string>& texts) = 0;
    
    // 获取向量维度
    virtual int GetDimension() const = 0;
};

// RAG 查询引擎
class RAGQueryEngine {
public:
    RAGQueryEngine(VectorStore& vector_store, EmbeddingService& embedding_service);
    
    // 添加文档到知识库
    bool AddDocument(const std::string& text, const std::string& metadata = "");
    
    // 批量添加文档
    bool AddDocuments(const std::vector<std::string>& texts);
    
    // 查询相似文档
    std::vector<SearchResult> Query(const std::string& query_text, int top_k = 5);
    
private:
    VectorStore& vector_store_;
    EmbeddingService& embedding_service_;
};

} // namespace rag
