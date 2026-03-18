#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <sstream>

#include "vector_store.h"
#include "sqlite_vec_extension.h"

// 生成随机向量（1024维，模拟真实嵌入）
std::vector<float> GenerateRandomVector(int dim) {
    std::vector<float> vec(dim);
    std::mt19937 gen(42); // 固定种子便于复现
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < dim; ++i) {
        vec[i] = dist(gen);
    }
    // 归一化（cosine similarity 需要单位向量）
    float norm = 0.0f;
    for (float v : vec) norm += v * v;
    norm = std::sqrt(norm);
    if (norm > 0) {
        for (float& v : vec) v /= norm;
    }
    return vec;
}

// 打印向量摘要
void PrintVectorSummary(const std::vector<float>& vec, int preview = 5) {
    std::cout << "[";
    for (int i = 0; i < std::min(preview, (int)vec.size()); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << std::fixed << std::setprecision(4) << vec[i];
    }
    std::cout << "...] (dim=" << vec.size() << ")";
}

// 获取当前时间字符串
std::string GetCurrentTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
    return ss.str();
}

// ==================== 演示1：基础向量操作 ====================
void DemoBasicOperations() {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "演示1: 基础向量存储与搜索" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    
    // 配置
    rag::VectorStoreConfig config;
    config.db_path = "demo_basic.db";
    config.table_name = "test_vectors";
    config.vector_dimension = 1024;
    config.distance_metric = "cosine";
    
    // 初始化
    rag::VectorStore store(config);
    if (!store.Initialize()) {
        std::cerr << "[ERROR] 初始化失败: " << store.GetLastError() << std::endl;
        return;
    }
    
    std::cout << "[SQLite3] VectorStore initialized: " << config.db_path << std::endl;
    
    // 生成并插入测试向量
    const int num_vectors = 10;
    std::cout << "[INFO] Inserting " << num_vectors << " vectors..." << std::endl;
    
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_vectors; ++i) {
        auto vec = GenerateRandomVector(1024);
        std::string content = "Test message " + std::to_string(i);
        store.InsertVector(-1, vec, content);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "[OK] Insert completed in " << duration << " ms" << std::endl;
    
    // 统计总数
    std::cout << "[OK] Total vectors: " << store.GetVectorCount() << std::endl;
    
    // 相似性搜索
    auto query_vec = GenerateRandomVector(1024);
    std::cout << "\n[QUERY] Query vector: ";
    PrintVectorSummary(query_vec);
    std::cout << std::endl;
    
    std::cout << "[INFO] Searching top-5 similar vectors..." << std::endl;
    auto results = store.SearchSimilar(query_vec, 5);
    
    std::cout << "\n--- Search Results ---" << std::endl;
    for (size_t i = 0; i < results.size(); ++i) {
        std::cout << "Rank " << (i + 1) << ": RowID=" << results[i].row_id 
                  << ", Distance=" << std::fixed << std::setprecision(6) << results[i].distance
                  << ", Content=" << results[i].content << std::endl;
    }
    
    std::cout << "\n[OK] Demo1 completed!" << std::endl;
}

// ==================== 演示2：元数据存储（IM消息场景） ====================
void DemoMetadataStorage() {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "演示2: 带元数据的向量存储（IM消息场景）" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    
    // 配置
    rag::VectorStoreConfig config;
    config.db_path = "demo_metadata.db";
    config.table_name = "msg_vectors";
    config.vector_dimension = 1024;
    config.distance_metric = "cosine";
    
    // 初始化
    rag::VectorStore store(config);
    if (!store.Initialize()) {
        std::cerr << "[ERROR] 初始化失败: " << store.GetLastError() << std::endl;
        return;
    }
    
    std::cout << "[SQLite3] VectorStore initialized: " << config.db_path << std::endl;
    
    // 模拟 IM 消息数据
    std::vector<std::pair<std::string, std::string>> messages = {
        {"conv_001", "明天下午3点开会讨论项目进度"},
        {"conv_001", "好的，我会准时参加"},
        {"conv_002", "这个bug怎么复现？"},
        {"conv_002", "在用户登录模块，点击忘记密码后触发"},
        {"conv_003", "周报内容：本周完成了搜索模块的开发"}
    };
    
    std::cout << "[INFO] Inserting " << messages.size() << " messages with metadata..." << std::endl;
    
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < messages.size(); ++i) {
        const auto& [conv_id, content] = messages[i];
        
        // 准备元数据
        rag::VectorMetadata meta;
        meta.convention_id = conv_id;
        meta.servermessage_id = "msg_" + std::to_string(1000 + i);
        meta.recordtype = "text";
        meta.orinaccout = "user_" + std::to_string(i % 3 + 1);
        meta.msgTimestamp = 1704067200 + i * 3600; // 模拟时间戳
        meta.content = content;
        meta.created_at = GetCurrentTimestamp();
        
        // 生成向量（模拟 embedding 结果）
        auto vec = GenerateRandomVector(1024);
        
        // 插入
        int64_t row_id = store.InsertVector(-1, vec, meta);
        if (row_id > 0) {
            std::cout << "  [Insert] RowID=" << row_id 
                      << ", Conv=" << conv_id 
                      << ", Content=\"" << content << "\"" << std::endl;
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "[OK] Insert completed in " << duration << " ms" << std::endl;
    
    // 统计总数
    std::cout << "[OK] Total vectors: " << store.GetVectorCount() << std::endl;
    
    // 相似性搜索
    auto query_vec = GenerateRandomVector(1024);
    std::cout << "\n[QUERY] Query vector: ";
    PrintVectorSummary(query_vec);
    std::cout << std::endl;
    
    std::cout << "[INFO] Searching top-5 similar messages..." << std::endl;
    auto results = store.SearchSimilar(query_vec, 5);
    
    std::cout << "\n--- Search Results ---" << std::endl;
    for (size_t i = 0; i < results.size(); ++i) {
        std::cout << "Rank " << (i + 1) << ": RowID=" << results[i].row_id 
                  << ", Distance=" << std::fixed << std::setprecision(6) << results[i].distance
                  << ", Content=" << results[i].content << std::endl;
    }
    
    std::cout << "\n[OK] Demo2 completed!" << std::endl;
}

// ==================== 演示3：扩展信息 ====================
void DemoExtensionInfo() {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "演示3: sqlite-vec 扩展信息" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    
    // 先加载一次扩展
    sqlite3* db;
    int rc = sqlite3_open(":memory:", &db);
    if (rc != SQLITE_OK) {
        std::cerr << "[ERROR] Failed to open memory DB" << std::endl;
        return;
    }
    
    std::string error_msg;
    bool loaded = rag::SqliteVecExtension::Load(db, "third_party/vec0.dll", error_msg);
    
    if (!loaded) {
        std::cerr << "[WARNING] Extension not loaded: " << error_msg << std::endl;
        std::cerr << "          检查 third_party/vec0.dll 是否存在" << std::endl;
    } else {
        std::cout << "[OK] sqlite-vec extension loaded successfully" << std::endl;
        
        if (rag::SqliteVecExtension::IsLoaded(db)) {
            std::string version = rag::SqliteVecExtension::GetVersion(db);
            std::cout << "[INFO] Extension version: " << version << std::endl;
        }
    }
    
    sqlite3_close(db);
}

// ==================== 主函数 ====================
int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  SQLite-Vec 向量搜索示例程序" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;
    
    std::cout << "系统要求:" << std::endl;
    std::cout << "  - SQLite3 (已集成)" << std::endl;
    std::cout << "  - sqlite-vec 扩展 (vec0.dll)" << std::endl;
    std::cout << "  - 向量维度: 1024 (qwen3-0.6b-embedding)" << std::endl;
    std::cout << std::endl;
    
    // 运行所有演示
    DemoExtensionInfo();
    DemoBasicOperations();
    DemoMetadataStorage();
    
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "所有演示已完成!" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    
    return 0;
}
