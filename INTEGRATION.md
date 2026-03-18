# native-emmbedding 集成指南

本文档介绍如何将 sqlite-vec 向量搜索项目与你的 native-emmbedding 项目集成。

## 集成架构

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   用户查询      │────▶│  native-emmbedding│────▶│   sqlite-vec    │
│  (自然语言)     │     │  (文本 → 向量)    │     │ (向量存储/检索) │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                          │
                                                          ▼
                                                 ┌─────────────────┐
                                                 │   xx助手 LLM    │
                                                 │  (生成回答)     │
                                                 └─────────────────┘
```

## 步骤1: 确定集成方式

### 方式A: DLL 导出接口 (推荐)

在 native-emmbedding 项目中添加导出接口：

```cpp
// native_embedding_api.h
#pragma once

#ifdef NATIVE_EMBEDDING_EXPORTS
#define EMBEDDING_API __declspec(dllexport)
#else
#define EMBEDDING_API __declspec(dllimport)
#endif

extern "C" {
    // 初始化 embedding 模型
    EMBEDDING_API int Embedding_Initialize(const char* model_path);
    
    // 文本转向量
    // text: 输入文本
    // output: 输出向量缓冲区 (需要预先分配 dim * sizeof(float) 字节)
    // dim: 向量维度 (如 1024)
    // 返回: 0 成功, 其他失败
    EMBEDDING_API int Embedding_Text(const char* text, float* output, int dim);
    
    // 批量嵌入
    EMBEDDING_API int Embedding_TextBatch(
        const char** texts, 
        int text_count,
        float* output,  // 输出缓冲区: text_count * dim
        int dim
    );
    
    // 获取向量维度
    EMBEDDING_API int Embedding_GetDimension();
    
    // 释放资源
    EMBEDDING_API void Embedding_Cleanup();
}
```

### 方式B: 直接链接

如果两个项目在同一个解决方案中，可以直接链接：

```cmake
# CMakeLists.txt
target_link_libraries(vector_search 
    PRIVATE 
        native_embedding  # 你的 native-emmbedding 库
)
```

## 步骤2: 实现 EmbeddingService 接口

### 方案A 实现 (DLL 方式)

```cpp
// native_embedding_service.h
#pragma once
#include "vector_store.h"
#include <windows.h>

namespace rag {

class NativeEmbeddingService : public EmbeddingService {
public:
    NativeEmbeddingService();
    ~NativeEmbeddingService();
    
    // 初始化模型
    bool Initialize(const std::string& model_path);
    
    // EmbeddingService 接口实现
    Vector Embed(const std::string& text) override;
    std::vector<Vector> EmbedBatch(const std::vector<std::string>& texts) override;
    int GetDimension() const override;

private:
    HMODULE dll_handle_ = nullptr;
    int dimension_ = 1024;
    
    // 函数指针
    using InitFunc = int (*)(const char*);
    using EmbedFunc = int (*)(const char*, float*, int);
    using EmbedBatchFunc = int (*)(const char**, int, float*, int);
    using GetDimFunc = int (*)();
    using CleanupFunc = void (*)();
    
    InitFunc fn_init_ = nullptr;
    EmbedFunc fn_embed_ = nullptr;
    EmbedBatchFunc fn_embed_batch_ = nullptr;
    GetDimFunc fn_get_dim_ = nullptr;
    CleanupFunc fn_cleanup_ = nullptr;
};

} // namespace rag
```

```cpp
// native_embedding_service.cpp
#include "native_embedding_service.h"
#include <iostream>

namespace rag {

NativeEmbeddingService::NativeEmbeddingService() {
    // 加载 DLL
    dll_handle_ = LoadLibraryA("native_embedding.dll");
    if (!dll_handle_) {
        std::cerr << "Failed to load native_embedding.dll" << std::endl;
        return;
    }
    
    // 获取函数地址
    fn_init_ = (InitFunc)GetProcAddress(dll_handle_, "Embedding_Initialize");
    fn_embed_ = (EmbedFunc)GetProcAddress(dll_handle_, "Embedding_Text");
    fn_embed_batch_ = (EmbedBatchFunc)GetProcAddress(dll_handle_, "Embedding_TextBatch");
    fn_get_dim_ = (GetDimFunc)GetProcAddress(dll_handle_, "Embedding_GetDimension");
    fn_cleanup_ = (CleanupFunc)GetProcAddress(dll_handle_, "Embedding_Cleanup");
    
    if (fn_get_dim_) {
        dimension_ = fn_get_dim_();
    }
}

NativeEmbeddingService::~NativeEmbeddingService() {
    if (fn_cleanup_) {
        fn_cleanup_();
    }
    if (dll_handle_) {
        FreeLibrary(dll_handle_);
    }
}

bool NativeEmbeddingService::Initialize(const std::string& model_path) {
    if (!fn_init_) return false;
    return fn_init_(model_path.c_str()) == 0;
}

Vector NativeEmbeddingService::Embed(const std::string& text) {
    Vector vec(dimension_);
    if (fn_embed_) {
        fn_embed_(text.c_str(), vec.data(), dimension_);
    }
    return vec;
}

std::vector<Vector> NativeEmbeddingService::EmbedBatch(
    const std::vector<std::string>& texts) {
    
    std::vector<Vector> results;
    results.reserve(texts.size());
    
    if (fn_embed_batch_) {
        // 使用批量接口
        std::vector<const char*> c_strs;
        for (const auto& t : texts) {
            c_strs.push_back(t.c_str());
        }
        
        std::vector<float> buffer(texts.size() * dimension_);
        fn_embed_batch_(c_strs.data(), texts.size(), buffer.data(), dimension_);
        
        for (size_t i = 0; i < texts.size(); ++i) {
            Vector vec(dimension_);
            memcpy(vec.data(), &buffer[i * dimension_], dimension_ * sizeof(float));
            results.push_back(std::move(vec));
        }
    } else {
        // 回退到单条处理
        for (const auto& text : texts) {
            results.push_back(Embed(text));
        }
    }
    
    return results;
}

int NativeEmbeddingService::GetDimension() const {
    return dimension_;
}

} // namespace rag
```

## 步骤3: 完整使用示例

```cpp
// main.cpp
#include "vector_store.h"
#include "native_embedding_service.h"
#include <iostream>

int main() {
    // 1. 初始化 Embedding 服务
    rag::NativeEmbeddingService embed_service;
    if (!embed_service.Initialize("models/qwen3-0.6b-embedding")) {
        std::cerr << "Failed to initialize embedding service" << std::endl;
        return 1;
    }
    
    std::cout << "Embedding dimension: " << embed_service.GetDimension() << std::endl;
    
    // 2. 初始化向量存储
    rag::VectorStoreConfig config;
    config.db_path = "im_messages.db";
    config.table_name = "message_vectors";
    config.vector_dimension = embed_service.GetDimension();  // 1024
    config.distance_metric = "cosine";
    
    rag::VectorStore store(config);
    if (!store.Initialize()) {
        std::cerr << "Failed to initialize vector store: " << store.GetLastError() << std::endl;
        return 1;
    }
    
    // 3. 创建 RAG 引擎
    rag::RAGQueryEngine rag(store, embed_service);
    
    // 4. 索引历史消息 (假设从 IM 数据库读取)
    std::vector<std::string> historical_messages = {
        "明天下午3点项目评审会议，请准时参加",
        "客户反馈登录功能有问题，需要紧急修复",
        "本周五团建活动，地点在西湖区",
        "新版本v2.1.0已经发布，请更新",
        "下周一开始全员远程办公"
    };
    
    std::cout << "Indexing " << historical_messages.size() << " messages..." << std::endl;
    for (const auto& msg : historical_messages) {
        rag.AddDocument(msg);
    }
    
    // 5. 用户查询
    std::string user_query = "什么时候开会？";
    std::cout << "\nUser query: " << user_query << std::endl;
    
    // 6. 检索相关消息
    auto results = rag.Query(user_query, 3);
    
    std::cout << "Retrieved " << results.size() << " relevant messages:" << std::endl;
    for (const auto& r : results) {
        std::cout << "  [distance=" << r.distance << "] " << r.content << std::endl;
    }
    
    // 7. 构造 Prompt 并调用 LLM
    std::string prompt = "基于以下历史消息回答问题:\n";
    for (const auto& r : results) {
        prompt += "- " + r.content + "\n";
    }
    prompt += "\n用户问题: " + user_query + "\n";
    prompt += "请根据以上信息回答问题。";
    
    // call_xx_assistant_llm(prompt);
    
    return 0;
}
```

## 步骤4: 编译配置

### 更新 CMakeLists.txt

```cmake
# 添加 native_embedding_service 到库
add_library(vector_search STATIC
    src/vector_store.cpp
    src/sqlite_vec_extension.cpp
    src/native_embedding_service.cpp  # 新增
)

target_include_directories(vector_search
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${SQLite3_INCLUDE_DIRS}
)

target_link_libraries(vector_search
    PUBLIC
        SQLite::SQLite3
)

# 主程序
add_executable(rag_system main.cpp)
target_link_libraries(rag_system PRIVATE vector_search)
```

## 步骤5: 运行测试

```bash
# 1. 确保 DLL 在 PATH 中或同一目录
set PATH=%PATH%;C:\path\to\native_embedding\bin

# 2. 运行程序
rag_system.exe
```

## 性能优化建议

1. **模型预热**: 首次调用 embedding 时模型需要加载，建议初始化后立即预热
2. **批处理**: 尽量使用 `EmbedBatch` 而非单条 `Embed`
3. **异步处理**: 消息入库可以异步进行，不影响用户体验
4. **增量索引**: 只索引新消息，不要全量重建

## 故障排除

### DLL 加载失败
- 检查 DLL 路径是否正确
- 确认所有依赖 DLL 都在同一目录
- 使用 Dependency Walker 检查依赖

### 向量维度不匹配
- 确认 `config.vector_dimension` 与模型输出一致
- qwen3-0.6b-embedding 输出维度为 1024

### 搜索结果不准确
- 检查向量是否归一化
- 确认使用 cosine 距离度量
- 尝试调整 top_k 参数

## 下一步

1. 实现 Agentic RAG 的检索策略
2. 添加消息的时间/会话过滤
3. 接入 xx助手 LLM 接口
4. 优化大规模数据性能
