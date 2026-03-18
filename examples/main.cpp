#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <sstream>
#include <iomanip>
#include <unordered_map>
#include <algorithm>

#include "vector_store.h"
#include "sqlite_vec_extension.h"
#include <sqlite3.h>

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

// 获取当前时间字符串
std::string GetCurrentTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
    return ss.str();
}

// 创建测试用的元数据
rag::VectorMetadata CreateTestMetadata(int index, const std::string& conv_id = "conv_test") {
    rag::VectorMetadata meta;
    meta.convention_id = conv_id;
    meta.servermessage_id = "msg_" + std::to_string(1000 + index);
    meta.recordtype = "text";
    meta.orinaccout = "user_" + std::to_string(index % 3 + 1);
    meta.msgTimestamp = 1704067200 + index * 3600;
    meta.content = "Test message " + std::to_string(index);
    meta.created_at = GetCurrentTimestamp();
    return meta;
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

// ==================== Demo 1: Basic Vector Operations ====================
void DemoBasicOperations() {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "Demo 1: Basic Vector Storage and Search" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    
    // Config
    rag::VectorStoreConfig config;
    config.db_path = "demo_basic.db";
    config.table_name = "test_vectors";
    config.vector_dimension = 1024;
    config.distance_metric = "cosine";
    
    // Initialize
    rag::VectorStore store(config);
    if (!store.Initialize()) {
        std::cerr << "[ERROR] Initialize failed: " << store.GetLastError() << std::endl;
        return;
    }
    
    std::cout << "[SQLite3] VectorStore initialized: " << config.db_path << std::endl;
    
    // Insert test vectors
    const int num_vectors = 10;
    std::cout << "[INFO] Inserting " << num_vectors << " vectors..." << std::endl;
    
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_vectors; ++i) {
        auto vec = GenerateRandomVector(1024);
        auto meta = CreateTestMetadata(i, "conv_demo1");
        store.InsertVector(-1, vec, meta);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "[OK] Insert completed in " << duration << " ms" << std::endl;
    
    // Count
    std::cout << "[OK] Total vectors: " << store.GetVectorCount() << std::endl;
    
    // Similarity search
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
    
    // Filter search demo
    std::cout << "\n[INFO] Testing filter search (convention_id = 'conv_demo1')..." << std::endl;
    std::unordered_map<std::string, std::string> filters;
    filters["convention_id"] = "conv_demo1";
    auto filter_results = store.SearchSimilarWithFilter(query_vec, 5, filters);
    
    std::cout << "--- Filter Search Results ---" << std::endl;
    std::cout << "Found " << filter_results.size() << " results with filter" << std::endl;
    for (size_t i = 0; i < filter_results.size(); ++i) {
        std::cout << "Rank " << (i + 1) << ": RowID=" << filter_results[i].row_id 
                  << ", Distance=" << std::fixed << std::setprecision(6) << filter_results[i].distance << std::endl;
    }
    
    std::cout << "\n[OK] Demo 1 completed!" << std::endl;
}

// ==================== Demo 2: Metadata Storage (IM Message Scene) ====================
void DemoMetadataStorage() {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "Demo 2: Vector Storage with Metadata (IM Message Scene)" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    
    // Config
    rag::VectorStoreConfig config;
    config.db_path = "demo_metadata.db";
    config.table_name = "msg_vectors";
    config.vector_dimension = 1024;
    config.distance_metric = "cosine";
    
    // Initialize
    rag::VectorStore store(config);
    if (!store.Initialize()) {
        std::cerr << "[ERROR] Initialize failed: " << store.GetLastError() << std::endl;
        return;
    }
    
    std::cout << "[SQLite3] VectorStore initialized: " << config.db_path << std::endl;
    
    // Simulate IM message data
    std::vector<std::pair<std::string, std::string>> messages = {
        {"conv_001", "Meeting at 3pm tomorrow to discuss project progress"},
        {"conv_001", "OK, I will attend on time"},
        {"conv_002", "How to reproduce this bug?"},
        {"conv_002", "In user login module, triggered after clicking forgot password"},
        {"conv_003", "Weekly report: completed search module development this week"}
    };
    
    std::cout << "[INFO] Inserting " << messages.size() << " messages with metadata..." << std::endl;
    
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < messages.size(); ++i) {
        const auto& [conv_id, content] = messages[i];
        
        // Prepare metadata
        rag::VectorMetadata meta;
        meta.convention_id = conv_id;
        meta.servermessage_id = "msg_" + std::to_string(1000 + i);
        meta.recordtype = "text";
        meta.orinaccout = "user_" + std::to_string(i % 3 + 1);
        meta.msgTimestamp = 1704067200 + i * 3600;
        meta.content = content;
        meta.created_at = GetCurrentTimestamp();
        
        // Generate vector (simulate embedding result)
        auto vec = GenerateRandomVector(1024);
        
        // Insert
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
    
    // Count
    std::cout << "[OK] Total vectors: " << store.GetVectorCount() << std::endl;
    
    // Similarity search
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
    
    // Filter search by conversation
    std::cout << "\n[INFO] Filter search (conv_001 only)..." << std::endl;
    std::unordered_map<std::string, std::string> filters;
    filters["convention_id"] = "conv_001";
    auto filter_results = store.SearchSimilarWithFilter(query_vec, 5, filters);
    
    std::cout << "--- Filter Search Results (conv_001) ---" << std::endl;
    std::cout << "Found " << filter_results.size() << " results" << std::endl;
    for (size_t i = 0; i < filter_results.size(); ++i) {
        std::cout << "Rank " << (i + 1) << ": RowID=" << filter_results[i].row_id 
                  << ", Distance=" << std::fixed << std::setprecision(6) << filter_results[i].distance << std::endl;
    }
    
    std::cout << "\n[OK] Demo 2 completed!" << std::endl;
}

// ==================== Demo 3: Extension Info ====================
void DemoExtensionInfo() {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "Demo 3: sqlite-vec Extension Info" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    
    // Load extension once
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
        std::cerr << "          Check if third_party/vec0.dll exists" << std::endl;
    } else {
        std::cout << "[OK] sqlite-vec extension loaded successfully" << std::endl;
        
        if (rag::SqliteVecExtension::IsLoaded(db)) {
            std::string version = rag::SqliteVecExtension::GetVersion(db);
            std::cout << "[INFO] Extension version: " << version << std::endl;
        }
    }
    
    sqlite3_close(db);
}

// ==================== Main Function ====================
int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  SQLite-Vec Vector Search Example" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;
    
    std::cout << "System Requirements:" << std::endl;
    std::cout << "  - SQLite3 (integrated)" << std::endl;
    std::cout << "  - sqlite-vec extension (vec0.dll)" << std::endl;
    std::cout << "  - Vector dimension: 1024 (qwen3-0.6b-embedding)" << std::endl;
    std::cout << std::endl;
    
    // Run all demos
    DemoExtensionInfo();
    DemoBasicOperations();
    DemoMetadataStorage();
    
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "All demos completed!" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    
    return 0;
}
