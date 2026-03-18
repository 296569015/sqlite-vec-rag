# SQLite-Vec 向量搜索项目

基于 sqlite-vec 的本地行向量搜索实现，用于企业即时通讯软件的本地 RAG 系统。

## 项目背景

本项目是消息本地 RAG 系统的核心组件，与 native-emmbedding 项目配合，实现：
- **native-emmbedding**: 文本 → 向量（qwen3-0.6b-embedding，输出维度 1024）
- **sqlite-vec**: 向量存储 + 相似性搜索
- **xx助手 LLM**: 生成最终回答（外部接入）

## 技术栈

- **C++17**: 核心实现语言
- **SQLite3**: 数据库存储（可选，支持内存后端演示）
- **sqlite-vec**: 向量搜索扩展
- **CMake**: 构建系统

## 快速开始

### 环境要求

- Windows 10/11 或 Linux/macOS
- Visual Studio 2019+ (Windows) 或 GCC/Clang (Linux/macOS)
- CMake 3.16+

### 构建项目

```bash
# 进入项目目录
cd sqlite-vec

# 创建构建目录
mkdir build && cd build

# 配置（使用内存后端，无需 SQLite3）
cmake .. -DUSE_MEMORY_BACKEND=ON

# 构建
cmake --build . --config Release
```

### 运行示例

```bash
# Windows
.\bin\Release\vector_search_example.exe

# Linux/macOS
./bin/vector_search_example
```

**示例输出：**

```
SQLite-Vec Vector Search Example
================================
Demonstrates local RAG system for IM software

========================================
Demo 1: Basic Vector Storage and Search
========================================

[Memory Backend] VectorStore initialized
[OK] VectorStore initialized

Inserting 10 messages...
  Inserted [1]: Meeting tomorrow at 3 PM...
  Inserted [2]: Please submit weekly work report...
  ...

[OK] Insert completed in 0 ms
[OK] Total vectors: 10

---------- Search Test ----------

Query: "When is the meeting"

========== Search Results ==========
Found 3 results

[1] RowID: 1, Distance: 0.977451
    Content: Meeting tomorrow at 3 PM to discuss project progress
[2] RowID: 2, Distance: 0.965969
    Content: Please submit weekly work report
...
```

## 项目结构

```
sqlite-vec/
├── CMakeLists.txt              # 构建配置
├── README.md                   # 本文档
├── INTEGRATION.md              # 与 native-emmbedding 集成指南
├── download_sqlite_vec.ps1     # 下载 sqlite-vec 扩展脚本
├── include/
│   ├── rag_engine.h            # 主入口头文件
│   └── vector_store.h          # 向量存储核心接口
├── src/
│   ├── vector_store.cpp        # 向量存储实现
│   └── sqlite_vec_extension.cpp # sqlite-vec 扩展加载器
└── examples/
    └── main.cpp                # 完整示例程序
```

## 核心 API

### VectorStore - 向量存储

```cpp
#include "vector_store.h"

// 1. 配置
rag::VectorStoreConfig config;
config.db_path = "messages.db";           // 数据库文件路径
config.table_name = "msg_vectors";        // 表名
config.vector_dimension = 1024;           // 向量维度（qwen3-0.6b 为 1024）
config.distance_metric = "cosine";        // 距离度量：cosine, l2

// 2. 初始化
rag::VectorStore store(config);
if (!store.Initialize()) {
    std::cerr << "Failed: " << store.GetLastError() << std::endl;
    return;
}

// 3. 插入向量（row_id = -1 表示自动分配）
int64_t row_id = store.InsertVector(-1, vector, "消息内容");

// 4. 相似性搜索
auto results = store.SearchSimilar(query_vector, top_k);

// 5. 处理结果
for (const auto& result : results) {
    std::cout << "RowID: " << result.row_id 
              << ", Distance: " << result.distance 
              << ", Content: " << result.content << std::endl;
}
```

### RAGQueryEngine - RAG 查询引擎

```cpp
// 创建 RAG 引擎（需要 EmbeddingService 实现）
rag::RAGQueryEngine rag(store, embed_service);

// 添加文档到知识库
rag.AddDocument("这是需要检索的文档内容");

// 查询
auto results = rag.Query("用户的查询问题", top_k);

// 使用检索结果作为上下文，调用 LLM 生成回答
```

### EmbeddingService 接口

你需要实现此接口以接入 native-emmbedding：

```cpp
class MyEmbeddingService : public rag::EmbeddingService {
public:
    // 将文本转换为向量
    rag::Vector Embed(const std::string& text) override {
        // TODO: 调用 native-emmbedding
        // 返回 1024 维的向量
    }
    
    // 批量嵌入
    std::vector<rag::Vector> EmbedBatch(const std::vector<std::string>& texts) override {
        // TODO: 批量嵌入实现
    }
    
    // 获取向量维度
    int GetDimension() const override { return 1024; }
};
```

## 与 native-emmbedding 集成

### 方案1: DLL 导出接口（推荐）

在 native-emmbedding 项目中添加导出接口：

```cpp
// native_embedding_api.h
extern "C" {
    __declspec(dllexport) int Embedding_Initialize(const char* model_path);
    __declspec(dllexport) int Embedding_Text(const char* text, float* output, int dim);
    __declspec(dllexport) int Embedding_GetDimension();
    __declspec(dllexport) void Embedding_Cleanup();
}
```

然后实现 `EmbeddingService` 接口调用这些函数（详见 `INTEGRATION.md`）。

### 方案2: 直接链接

如果两个项目在同一解决方案中，直接链接 native-emmbedding 库：

```cmake
target_link_libraries(your_app PRIVATE native_embedding)
```

## 构建选项

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `USE_MEMORY_BACKEND` | `ON` | 使用内存后端（无需 SQLite3） |
| `BUILD_EXAMPLES` | `ON` | 构建示例程序 |

### 使用 SQLite3 持久化存储

```bash
# Windows - 需要 SQLite3 开发库
vcpkg install sqlite3

# 配置
cmake .. -DUSE_MEMORY_BACKEND=OFF

# 构建
cmake --build . --config Release
```

## sqlite-vec 扩展

### 方式1: 加载扩展 DLL（推荐用于生产）

1. 下载 sqlite-vec 扩展：

```powershell
.\download_sqlite_vec.ps1
```

2. 将 `vec0.dll` 放到可执行目录

3. 代码自动加载扩展

### 方式2: 纯 SQL 回退实现

如果无法加载扩展，代码会自动使用纯 SQL 实现，但性能会下降（全表扫描）。

## 性能数据

在典型开发机上测试（内存后端）：

| 操作 | 性能 |
|------|------|
| 向量插入 | ~100,000+ 条/秒 |
| 相似性搜索 | ~10-100 μs（100 条数据） |
| 批量嵌入 | 取决于 native-emmbedding |

**注意：** 实际性能取决于向量维度、数据量和硬件配置。

## 示例场景

### 场景1: 消息历史搜索

```cpp
// 用户输入搜索关键词
std::string query = "上周的会议纪要";

// 转换为向量
auto query_vec = embed_service.Embed(query);

// 搜索相似消息
auto results = store.SearchSimilar(query_vec, 10);

// 展示结果给用户
for (const auto& r : results) {
    display_message(r.row_id, r.content, r.distance);
}
```

### 场景2: 智能问答（RAG）

```cpp
// 构建知识库
for (const auto& msg : important_messages) {
    rag_engine.AddDocument(msg);
}

// 用户提问
std::string question = "这个需求什么时候截止？";
auto relevant_msgs = rag_engine.Query(question, 5);

// 构造 Prompt
std::string prompt = "基于以下历史消息回答问题:\n";
for (const auto& msg : relevant_msgs) {
    prompt += "- " + msg.content + "\n";
}
prompt += "\n问题: " + question + "\n回答:";

// 调用 xx助手 LLM 生成回答
std::string answer = call_xx_assistant_llm(prompt);
```

## 配置说明

### VectorStoreConfig 参数

```cpp
struct VectorStoreConfig {
    std::string db_path = "vector_store.db";     // 数据库文件路径
    std::string table_name = "vectors";           // 表名前缀
    int vector_dimension = 1024;                  // 向量维度
    std::string distance_metric = "cosine";       // 距离度量
    bool use_index = true;                        // 是否使用索引（sqlite-vec）
};
```

## 常见问题

### Q: 编译时出现编码警告？

A: 这是 MSVC 对 UTF-8 字符的警告，不影响功能。可以在 CMakeLists.txt 中添加：

```cmake
add_compile_options(/source-charset:utf-8)
```

### Q: sqlite-vec 扩展加载失败？

A: 确保 `vec0.dll` 在可执行文件的同一目录或系统 PATH 中。也可以修改 `SqliteVecExtension::Load` 中的搜索路径。

### Q: 向量维度不匹配？

A: 检查 `VectorStoreConfig::vector_dimension` 是否与 embedding 模型输出一致：
- qwen3-0.6b-embedding: 1024 维
- 其他模型请参考模型文档

### Q: 如何清理数据库？

A: 
```cpp
// 代码中清理
store.ClearAll();

// 或直接删除文件
std::remove("messages.db");
```

### Q: 支持并发访问吗？

A: 
- **内存后端**: 支持多线程读，单线程写（已加锁）
- **SQLite3**: 支持多进程并发访问（SQLite3 特性）

## 后续开发路线图

1. **Agentic RAG**: 实现更智能的检索策略
   - 查询重写（Query Rewriting）
   - 多轮检索
   - 结果重排序

2. **增量更新**: 支持消息的增量索引

3. **量化压缩**: FP16/INT8 量化减少存储空间

4. **多租户**: 支持不同用户/群组的隔离存储

5. **元数据过滤**: 支持时间范围、会话 ID 等过滤条件

6. **混合搜索**: 结合关键词搜索和向量搜索

## 调试技巧

### 启用日志输出

```cpp
// 在 VectorStore 初始化前设置
std::cout << "[Debug] Initializing VectorStore..." << std::endl;
```

### 检查向量数据

```cpp
// 打印向量前 10 个元素
void PrintVector(const rag::Vector& vec) {
    std::cout << "Vector [" << vec.size() << "D]: ";
    for (int i = 0; i < std::min(10, (int)vec.size()); ++i) {
        std::cout << vec[i] << " ";
    }
    std::cout << "..." << std::endl;
}
```

### 性能分析

```cpp
auto start = std::chrono::high_resolution_clock::now();
auto results = store.SearchSimilar(query_vec, 10);
auto end = std::chrono::high_resolution_clock::now();
auto us = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
std::cout << "Search took " << us.count() << " μs" << std::endl;
```

## 参考链接

- [sqlite-vec 官方文档](https://alexgarcia.xyz/sqlite-vec/)
- [SQLite3 C/C++ API](https://www.sqlite.org/capi3.html)
- [Qwen3 Embedding 模型](https://huggingface.co/Qwen)
- [RAG 论文](https://arxiv.org/abs/2005.11401)

## 贡献指南

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送分支 (`git push origin feature/AmazingFeature`)
5. 创建 Pull Request

## 许可证

MIT License - 详见 LICENSE 文件

## 联系方式

- 项目维护者：小 C
- 部门：PC 客户端开发组
- 相关项目：native-emmbedding, xx助手
