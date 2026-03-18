# SQLite-Vec 向量搜索项目

基于 sqlite-vec 的本地行向量搜索实现，用于企业即时通讯软件的本地 RAG 系统。

## 项目背景

本项目是消息本地 RAG 系统的核心组件，与 native-emmbedding 项目配合，实现：
- **native-emmbedding**: 文本 → 向量（qwen3-0.6b-embedding，输出维度 1024）
- **sqlite-vec**: 向量存储 + 相似性搜索（通过 vec0.dll 扩展）
- **xx助手 LLM**: 生成最终回答（外部接入）

## 技术栈

- **C++17**: 核心实现语言
- **SQLite3**: 数据库存储（使用合并版本，源码已包含）
- **sqlite-vec**: 向量搜索扩展（通过 vec0.dll 加载）
- **CMake**: 构建系统

## 项目结构

```
sqlite-vec/
├── CMakeLists.txt              # 构建配置
├── README.md                   # 本文档
├── INTEGRATION.md              # 与 native-emmbedding 集成指南
├── LICENSE                     # MIT 许可证
├── sqlite3/                    # SQLite3 源码（已包含）
│   ├── sqlite3.c
│   └── sqlite3.h
├── third_party/                # 第三方扩展
│   └── vec0.dll                # sqlite-vec 扩展（需要自行下载）
├── include/
│   ├── rag_engine.h            # 主入口头文件
│   └── vector_store.h          # 向量存储核心接口
├── src/
│   ├── vector_store.cpp        # 向量存储实现（单表设计）
│   └── sqlite_vec_extension.cpp # sqlite-vec 扩展加载器
└── examples/
    └── main.cpp                # 完整示例程序
```

## 快速开始

### 环境要求

- Windows 10/11 或 Linux/macOS
- Visual Studio 2019+ (Windows) 或 GCC/Clang (Linux/macOS)
- CMake 3.16+

### 1. 下载 sqlite-vec 扩展

**Windows 用户：**
```powershell
# 方法1：使用脚本下载
.\download_sqlite_vec.ps1

# 方法2：手动下载
# 访问 https://github.com/asg017/sqlite-vec/releases
# 下载 vec0.dll 放到 third_party/ 目录
```

**注意**：`third_party/vec0.dll` 是可选的，但强烈推荐使用。如果没有，程序会自动回退到纯 SQL 实现（性能较差）。

### 2. 构建项目

```bash
# 进入项目目录
cd sqlite-vec

# 创建构建目录
mkdir build && cd build

# 配置（默认使用 SQLite3 后端）
cmake ..

# 构建
cmake --build . --config Release
```

### 3. 运行示例

```bash
# Windows
.\bin\Release\vector_search_example.exe

# Linux/macOS
./bin/vector_search_example
```

**预期输出：**
```
[SQLite-Vec] Extension loaded successfully
[SQLite3 Backend] VectorStore initialized: demo_basic.db (Extension: YES)
[OK] Insert completed in X ms
[OK] Total vectors: 10
...
All demos completed successfully!
```

运行后会生成 `.db` 文件，可以用 [DB Browser for SQLite](https://sqlitebrowser.org/) 查看。

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

## 数据库设计

### 单表结构

```sql
CREATE TABLE messages (
    rowid INTEGER PRIMARY KEY,      -- 消息 ID
    content TEXT,                    -- 消息文本内容
    embedding BLOB,                  -- 向量数据（1024 个 float）
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**设计说明**：
- 单表设计，同时存储文本和向量
- 向量以 BLOB 形式存储（1024 维 × 4 字节 = 4KB）
- 如果加载了 vec0.dll，会使用 `vec_distance_cosine()` 函数加速搜索

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
| `USE_SQLITE3` | `ON` | 使用 SQLite3 后端（需要 sqlite3.c/h） |
| `BUILD_EXAMPLES` | `ON` | 构建示例程序 |

### 切换后端

```bash
# 使用内存后端（无需 SQLite3）
cmake .. -DUSE_SQLITE3=OFF
```

## sqlite-vec 扩展

### 作用
- 提供 `vec_distance_cosine()` 等向量距离函数
- 支持虚拟表 `vec0` 进行高效索引（未来扩展）
- 显著提升搜索性能（3-10 倍）

### 加载机制
程序启动时会自动尝试加载 `third_party/vec0.dll`：
1. 成功 → 使用 `vec_distance_cosine()` 函数计算距离
2. 失败 → 回退到纯 SQL 实现（内存中计算距离）

### 下载地址
- GitHub: https://github.com/asg017/sqlite-vec/releases
- 选择 `sqlite-vec-vX.X.X-loadable-windows-x64.dll` 重命名为 `vec0.dll`

## 性能数据

在典型开发机上测试（100 条向量）：

| 配置 | 搜索时间 | 说明 |
|------|---------|------|
| 有 vec0.dll | ~500 μs | 使用原生距离函数 ✅ |
| 无 vec0.dll | ~1800 μs | 纯 SQL 实现 |
| 内存后端 | ~70 μs | 无持久化，最快 |

**注意**：实际性能取决于向量维度、数据量和硬件配置。

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

## 常见问题

### Q: vec0.dll 加载失败？

A: 确保 `vec0.dll` 在 `third_party/` 目录下，或与可执行文件在同一目录。程序会自动尝试多个路径加载。

### Q: 找不到 sqlite3.c？

A: 确保项目根目录下有 `sqlite3/sqlite3.c` 和 `sqlite3/sqlite3.h` 文件。这些文件已包含在仓库中。

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

A: SQLite3 支持多进程并发访问。如果启用 WAL 模式，读写性能会更好。

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
