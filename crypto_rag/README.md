
# Crypto RAG System

## 项目简介

Crypto RAG System 是一个基于检索增强生成（RAG）的工具，用于管理数字货币相关文档并回答用户查询。项目结合本地 LLM（DeepSeek-R1:7B）、PostgreSQL（带 pgvector 扩展）作为向量数据库，并通过 Gradio 提供用户界面。

## 功能

- **文档验证**：判断上传文档是否与数字货币相关。
- **文档添加**：将通过验证的文档存储到向量数据库。
- **回答生成**：根据用户查询检索相关文档并生成回答。

## 外部环境配置

### 安装 PostgreSQL 并启用 pgvector

#### Ubuntu:
```bash
sudo apt install postgresql postgresql-contrib
```

#### macOS:
```bash
brew install postgresql
```

#### Windows:
从 PostgreSQL 官网下载安装程序。

### 启动 PostgreSQL 服务

#### Ubuntu:
```bash
sudo service postgresql start
```

#### macOS:
```bash
brew services start postgresql
```

#### Windows:
通过服务管理器启动。

### 创建数据库并启用 pgvector：

```sql
CREATE DATABASE crypto_rag_db;
\c crypto_rag_db
CREATE EXTENSION vector;
```

### 安装和启动 Ollama

#### 安装 Ollama：

##### Linux/macOS:
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

##### Windows:
从 Ollama 官网下载安装程序。

### 下载并运行模型：

```bash
ollama pull deepseek-r1:7b
ollama run deepseek-r1:7b
```

## 依赖安装

确保已安装 Python 3.8+，然后运行：

```bash
pip install -r requirements.txt
```

## 运行程序

### 配置

编辑 `config.json`，填入数据库和模型信息。

### 启动

运行以下命令，然后访问 http://0.0.0.0:7860：

```bash
python main.py
```

## 前端界面说明

项目使用 Gradio 提供交互式前端界面，包含以下功能：

### Add Documents：
- **输入**：用户可以通过界面上传文档（支持常见格式，如 .txt、.pdf 等）。
- **输出**：系统会验证文档是否与数字货币相关，并显示验证结果（例如“文档有效，已存储”或“文档无效，未存储”）。
- **用途**：将通过验证的文档存储到 PostgreSQL 向量数据库。

### Retrieve：
- **输入**：用户在文本框中输入查询问题，例如“比特币的价格趋势是什么？”。
- **输出**：系统从向量数据库检索相关文档，并基于检索结果生成自然语言回答，显示在界面上。
- **用途**：提供基于文档内容的智能回答。

## 文件结构

- `config.json`：配置文件，包含数据库连接信息和模型参数。
- `main.py`：主程序入口，启动 Gradio 界面和 RAG 系统。
- `rag.py`：核心 RAG 逻辑，处理文档验证、存储和查询生成。
- `requirements.txt`：Python 依赖清单。

## 注意事项

- 确保 PostgreSQL 和 Ollama 服务在运行程序前已正确启动。
- 日志记录在 `crypto_rag.log` 文件中，可用于调试和监控。
