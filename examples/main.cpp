// Vector Search Example Program
// Demonstrates local RAG system for IM software

#include "vector_store.h"
#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <chrono>

using namespace rag;

// Mock Embedding Service - replace with native-emmbedding in production
class MockEmbeddingService : public EmbeddingService {
public:
    MockEmbeddingService(int dim = 1024) : dimension_(dim) {
        rng_.seed(42);
    }
    
    Vector Embed(const std::string& text) override {
        Vector vec(dimension_);
        
        size_t hash = std::hash<std::string>{}(text);
        std::mt19937 local_rng(hash);
        std::normal_distribution<float> dist(0.0f, 0.1f);
        
        for (int i = 0; i < dimension_; ++i) {
            vec[i] = dist(local_rng);
        }
        
        float norm = 0.0f;
        for (float v : vec) {
            norm += v * v;
        }
        norm = std::sqrt(norm);
        if (norm > 0) {
            for (auto& v : vec) {
                v /= norm;
            }
        }
        
        return vec;
    }
    
    std::vector<Vector> EmbedBatch(const std::vector<std::string>& texts) override {
        std::vector<Vector> results;
        results.reserve(texts.size());
        for (const auto& text : texts) {
            results.push_back(Embed(text));
        }
        return results;
    }
    
    int GetDimension() const override {
        return dimension_;
    }
    
private:
    int dimension_;
    std::mt19937 rng_;
};

// Print search results
void PrintResults(const std::vector<SearchResult>& results) {
    std::cout << "\n========== Search Results ==========" << std::endl;
    std::cout << "Found " << results.size() << " results\n" << std::endl;
    
    for (size_t i = 0; i < results.size(); ++i) {
        std::cout << "[" << (i + 1) << "] RowID: " << results[i].row_id 
                  << ", Distance: " << results[i].distance << std::endl;
        if (!results[i].content.empty()) {
            std::cout << "    Content: " << results[i].content.substr(0, 100);
            if (results[i].content.length() > 100) {
                std::cout << "...";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
}

// Demo 1: Basic vector storage and search
void DemoBasicUsage() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Demo 1: Basic Vector Storage and Search" << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    VectorStoreConfig config;
    config.db_path = "demo_basic.db";
    config.table_name = "messages";
    config.vector_dimension = 1024;
    config.distance_metric = "cosine";
    
    VectorStore store(config);
    if (!store.Initialize()) {
        std::cerr << "Failed to initialize: " << store.GetLastError() << std::endl;
        return;
    }
    
    std::cout << "[OK] VectorStore initialized" << std::endl;
    
    MockEmbeddingService embed_service(1024);
    
    // Sample IM messages
    std::vector<std::string> messages = {
        "Meeting tomorrow at 3 PM to discuss project progress",
        "Please submit weekly work report",
        "Company team building event next Saturday",
        "This bug needs urgent fix, affects user login",
        "New feature deployed, please test",
        "Tech sharing session this afternoon about AI RAG",
        "Code review this Friday afternoon",
        "Customer reported performance issue, needs optimization",
        "Next month version plan is ready",
        "Please update to latest client version"
    };
    
    std::cout << "\nInserting " << messages.size() << " messages..." << std::endl;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (size_t i = 0; i < messages.size(); ++i) {
        auto vec = embed_service.Embed(messages[i]);
        int64_t row_id = store.InsertVector(-1, vec, messages[i]);
        if (row_id < 0) {
            std::cerr << "Insert failed: " << store.GetLastError() << std::endl;
        } else {
            std::cout << "  Inserted [" << row_id << "]: " 
                      << messages[i].substr(0, 30) << "..." << std::endl;
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "\n[OK] Insert completed in " << duration.count() << " ms" << std::endl;
    std::cout << "[OK] Total vectors: " << store.GetVectorCount() << std::endl;
    
    // Search test
    std::cout << "\n---------- Search Test ----------" << std::endl;
    
    std::vector<std::string> queries = {
        "When is the meeting",
        "System has problems to fix",
        "Version update related"
    };
    
    for (const auto& query : queries) {
        std::cout << "\nQuery: \"" << query << "\"" << std::endl;
        
        auto query_vec = embed_service.Embed(query);
        auto results = store.SearchSimilar(query_vec, 3);
        
        PrintResults(results);
    }
}

// Demo 2: Batch insert and performance test
void DemoBatchInsert() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Demo 2: Batch Insert and Performance" << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    VectorStoreConfig config;
    config.db_path = "demo_batch.db";
    config.table_name = "batch_messages";
    config.vector_dimension = 1024;
    
    VectorStore store(config);
    if (!store.Initialize()) {
        std::cerr << "Failed to initialize: " << store.GetLastError() << std::endl;
        return;
    }
    
    MockEmbeddingService embed_service(1024);
    
    const int num_messages = 100;
    std::vector<std::string> batch_messages;
    batch_messages.reserve(num_messages);
    
    std::vector<std::string> templates = {
        "Notice about {}",
        "{} project progress update",
        "Meeting minutes for {}",
        "Solution for {} issue",
        "Schedule for next week {}"
    };
    
    std::vector<std::string> topics = {
        "tech sharing", "product review", "team building", "customer requirements", 
        "bug fix", "feature development", "performance optimization", "security audit"
    };
    
    for (int i = 0; i < num_messages; ++i) {
        std::string msg = templates[i % templates.size()];
        size_t pos = msg.find("{}");
        if (pos != std::string::npos) {
            msg.replace(pos, 2, topics[i % topics.size()]);
        }
        msg += " - Message " + std::to_string(i + 1);
        batch_messages.push_back(msg);
    }
    
    std::cout << "Embedding " << num_messages << " messages..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    auto vectors = embed_service.EmbedBatch(batch_messages);
    auto embed_end = std::chrono::high_resolution_clock::now();
    
    std::cout << "Inserting vectors..." << std::endl;
    std::vector<std::pair<Vector, std::string>> data;
    for (size_t i = 0; i < batch_messages.size(); ++i) {
        data.emplace_back(std::move(vectors[i]), batch_messages[i]);
    }
    
    bool success = store.InsertVectors(data);
    auto insert_end = std::chrono::high_resolution_clock::now();
    
    if (!success) {
        std::cerr << "Batch insert failed: " << store.GetLastError() << std::endl;
        return;
    }
    
    auto embed_time = std::chrono::duration_cast<std::chrono::milliseconds>(embed_end - start);
    auto insert_time = std::chrono::duration_cast<std::chrono::milliseconds>(insert_end - embed_end);
    
    std::cout << "\n[OK] Batch operation:" << std::endl;
    std::cout << "  - Embedding: " << embed_time.count() << " ms (" 
              << (num_messages * 1000.0 / embed_time.count()) << " msg/sec)" << std::endl;
    std::cout << "  - Insert: " << insert_time.count() << " ms (" 
              << (num_messages * 1000.0 / insert_time.count()) << " msg/sec)" << std::endl;
    std::cout << "  - Total vectors: " << store.GetVectorCount() << std::endl;
    
    // Search performance test
    std::cout << "\n---------- Search Performance ----------" << std::endl;
    std::string query = "Tech sharing schedule";
    auto query_vec = embed_service.Embed(query);
    
    auto search_start = std::chrono::high_resolution_clock::now();
    auto results = store.SearchSimilar(query_vec, 5);
    auto search_end = std::chrono::high_resolution_clock::now();
    
    auto search_time = std::chrono::duration_cast<std::chrono::microseconds>(search_end - search_start);
    
    std::cout << "\nQuery: \"" << query << "\"" << std::endl;
    std::cout << "Search time: " << search_time.count() << " us" << std::endl;
    PrintResults(results);
}

// Demo 3: RAG Query Engine
void DemoRAGQuery() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Demo 3: RAG Query Engine" << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    VectorStoreConfig config;
    config.db_path = "demo_rag.db";
    config.table_name = "knowledge_base";
    config.vector_dimension = 1024;
    
    VectorStore store(config);
    if (!store.Initialize()) {
        std::cerr << "Failed to initialize: " << store.GetLastError() << std::endl;
        return;
    }
    
    MockEmbeddingService embed_service(1024);
    RAGQueryEngine rag_engine(store, embed_service);
    
    // Build knowledge base
    std::vector<std::string> knowledge = {
        "SQLite-Vec is a vector search extension for SQLite, supporting efficient similarity search.",
        "RAG (Retrieval-Augmented Generation) combines retrieval and generation AI technology.",
        "Embedding converts text into high-dimensional vectors for semantic understanding.",
        "Qwen3-0.6B is a lightweight LLM by Alibaba Cloud, suitable for local deployment.",
        "Vector databases store and retrieve high-dimensional vectors with ANN search.",
        "Local RAG systems in IM software enable offline intelligent Q&A.",
        "Cosine similarity measures angle between vectors, commonly used for text similarity.",
        "Agentic RAG uses AI Agents for smarter retrieval-augmented generation."
    };
    
    std::cout << "Building knowledge base..." << std::endl;
    for (const auto& doc : knowledge) {
        if (!rag_engine.AddDocument(doc)) {
            std::cerr << "Failed to add document" << std::endl;
        }
    }
    std::cout << "[OK] Knowledge base built with " << store.GetVectorCount() << " documents\n" << std::endl;
    
    // RAG queries
    std::vector<std::string> queries = {
        "What is vector search?",
        "What are the advantages of local RAG?",
        "How to calculate text similarity?"
    };
    
    for (const auto& query : queries) {
        std::cout << "========================================" << std::endl;
        std::cout << "User Query: \"" << query << "\"" << std::endl;
        std::cout << "========================================" << std::endl;
        
        auto results = rag_engine.Query(query, 3);
        
        std::cout << "\nRetrieved relevant knowledge:" << std::endl;
        for (size_t i = 0; i < results.size(); ++i) {
            std::cout << "[" << (i + 1) << "] (similarity: " 
                      << (1.0f - results[i].distance) << ") "
                      << results[i].content << std::endl;
        }
        
        std::cout << "\n[LLM would generate answer using retrieved context...]" << std::endl;
        std::cout << "(In production, this would call xx-assistant LLM API with context)\n" << std::endl;
    }
}

int main() {
    std::cout << "SQLite-Vec Vector Search Example" << std::endl;
    std::cout << "================================" << std::endl;
    std::cout << "Demonstrates local RAG system for IM software\n" << std::endl;
    
    try {
        DemoBasicUsage();
        DemoBatchInsert();
        DemoRAGQuery();
        
        std::cout << "\n========================================" << std::endl;
        std::cout << "All demos completed successfully!" << std::endl;
        std::cout << "========================================" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
