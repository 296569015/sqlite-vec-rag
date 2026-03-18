# SQLite3 后端设置指南

## 方法1: 手动下载（推荐）

### 步骤1: 下载 SQLite3
1. 访问 https://www.sqlite.org/download.html
2. 找到 **SQLite Amalgamation** 部分
3. 下载 `sqlite-amalgamation-3450200.zip`
4. 解压到任意目录

### 步骤2: 复制文件
将解压后的以下两个文件复制到项目目录：
```
sqlite-vec/
└── build/
    └── sqlite3/
        ├── sqlite3.c    <-- 复制到这里
        └── sqlite3.h    <-- 复制到这里
```

### 步骤3: 重新构建
```powershell
cd c:\code\AI_RAG_system\sqlite-vec\build

# 清理旧构建
Remove-Item * -Recurse -Force

# 重新配置（启用 SQLite3）
cmake .. -DUSE_SQLITE3=ON

# 构建
cmake --build . --config Release
```

### 步骤4: 运行
```powershell
.\bin\Release\vector_search_example.exe
```

运行后会生成以下 db 文件：
- `demo_basic.db` - 基础示例数据库
- `demo_batch.db` - 批量插入示例数据库  
- `demo_rag.db` - RAG 示例数据库

可以用 [DB Browser for SQLite](https://sqlitebrowser.org/) 查看这些文件。

---

## 方法2: 使用 vcpkg（如果你已安装）

```powershell
# 安装 SQLite3
vcpkg install sqlite3:x64-windows

# 配置时指定 vcpkg 工具链
cmake .. -DUSE_SQLITE3=ON -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake
```

---

## 方法3: 使用预编译的 SQLite3 DLL

### 下载 DLL 版本
1. 访问 https://www.sqlite.org/download.html
2. 下载 **64-bit DLL (x64) for SQLite** (`sqlite-dll-win64-x64-3450200.zip`)
3. 解压得到 `sqlite3.dll` 和 `sqlite3.def`

### 创建导入库
```powershell
# 使用 lib.exe 创建导入库（在 VS 开发者命令提示符中）
lib /DEF:sqlite3.def /OUT:sqlite3.lib /MACHINE:x64
```

### 配置项目
```powershell
cmake .. -DUSE_SQLITE3=ON -DSQLite3_INCLUDE_DIR=C:/path/to/sqlite3 -DSQLite3_LIBRARY=C:/path/to/sqlite3.lib
```

---

## 验证安装

运行程序后，检查是否生成了 .db 文件：

```powershell
ls *.db

# 应该看到：
# demo_basic.db
# demo_batch.db
# demo_rag.db
```

使用 sqlite3 命令行查看内容：

```powershell
# 进入交互式 shell
sqlite3 demo_basic.db

# 查看表
.tables

# 查看内容
SELECT * FROM messages_content LIMIT 5;

# 退出
.quit
```

---

## 常见问题

### Q: 找不到 sqlite3.h
A: 确保文件在 `build/sqlite3/` 目录下，或修改 `SQLITE3_DIR` 路径：
```powershell
cmake .. -DUSE_SQLITE3=ON -DSQLITE3_DIR=C:/your/path/to/sqlite3
```

### Q: 链接错误
A: 确保使用的是相同架构（x64）的 SQLite3 库。

### Q: 运行时提示缺少 sqlite3.dll
A: 将 `sqlite3.dll` 复制到可执行文件目录，或添加到 PATH。

---

## 当前状态

你现在使用的是**内存后端**，特点：
- ✅ 无需额外配置，即编即用
- ✅ 性能更快
- ❌ 数据不持久化

切换到 **SQLite3 后端** 后：
- ✅ 数据持久化到 .db 文件
- ✅ 可用标准 SQLite 工具查看
- ✅ 支持事务和并发
