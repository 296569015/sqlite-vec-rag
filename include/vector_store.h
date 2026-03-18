#pragma once

#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <optional>

namespace rag {

// 向量类型定义
using Vector = std::vector<float>;

// 搜索结果结构
struct SearchResult {
    int64_t row_id;
    float distance;
    std::string content;  // 可选：原始内容
    
    SearchResult(int64_t id, float dist, std::string text = "")
        : row_id(id), distance(dist), content(std::move(text)) {}
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
    
    // 插入向量
    // @param row_id: 行ID（如果为-1则自动分配）
    // @param vector: 向量数据
    // @param content: 关联的原始文本内容（可选）
    // @return: 实际的 row_id，失败返回 -1
    int64_t InsertVector(int64_t row_id, const Vector& vector, const std::string& content = "");
    
    // 批量插入向量
    bool InsertVectors(const std::vector<std::pair<Vector, std::string>>& vectors);
    
    // 相似性搜索
    // @param query_vector: 查询向量
    // @param top_k: 返回最相似的 k 个结果
    // @return: 搜索结果列表
    std::vector<SearchResult> SearchSimilar(const Vector& query_vector, int top_k = 5);
    
    // 带过滤条件的搜索
    std::vector<SearchResult> SearchSimilarWithFilter(
        const Vector& query_vector, 
        int top_k, 
        const std::string& where_clause
    );
    
    // 删除向量
    bool DeleteVector(int64_t row_id);
    
    // 更新向量
    bool UpdateVector(int64_t row_id, const Vector& vector, const std::string& content = "");
    
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
    
    // 回退实现
    std::vector<SearchResult> SearchSimilarFallback(const Vector& query_vector, int top_k);
    static float CalculateCosineDistance(const float* a, const float* b, int dim);
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
