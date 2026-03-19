# RAG知识问答系统

基于检索增强生成（RAG）架构的专业知识问答系统，专注于计算机科学领域，支持中文专业书籍的智能化问答。

## 系统架构

### 1. 知识库构建

#### PDF解析层
- **PyPDFLoader**: LangChain内置的PDF文档加载器
- 自动提取PDF文本内容

#### 文本分块
- **RecursiveCharacterTextSplitter**: LangChain提供的递归文本分割器
- **Chunk大小**: 512字符，Overlap: 50字符
- 支持中文分隔符（。！？等）

### 2. 检索层

#### 向量存储
- **FAISS**: 高效的向量相似度搜索库
- **HuggingFaceEmbeddings**: all-MiniLM-L6-v2 嵌入模型（384维）

#### 检索机制
- **余弦相似度**: 计算查询与文档片段的语义匹配度
- **Top-K召回**: 返回最相关的K个知识片段

### 3. 评估层

#### 相似度阈值过滤
- **阈值**: 0.7（可配置）
- 自动拒绝低相关度检索结果，减少噪声干扰

#### 证据对齐
- 在前端高亮标注原文来源
- 实现可追溯的答案生成，增强可信度

### 4. 大语言模型集成

使用LangChain统一接口：
- **Qwen (千问)**: 阿里云通义千问系列
- **Claude**: Anthropic Claude系列

### 5. 可视化应用

基于Streamlit构建的Web交互界面：
- 问答输入与结果显示
- 检索原文高亮展示
- 相似度分数可视化

## 快速开始

### 环境配置

```bash
# 克隆项目
git clone https://github.com/wicode1025/RAGknowledge.git
cd RAGknowledge

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 配置API密钥

编辑 `config.py`：

```python
# Qwen配置
QWEN_API_KEY = "your-qwen-api-key"

# 选择LLM提供商 (qwen 或 claude)
LLM_PROVIDER = "qwen"
```

### 构建知识库

```bash
python main_langchain.py
```

### 启动Web界面

```bash
streamlit run visualization/langchain_app.py
```

## 项目结构

```
RAGknowledge/
├── langchain_rag.py              # 核心RAG系统（LangChain实现）
├── main_langchain.py             # 终端入口脚本
├── config.py                     # 配置文件
├── visualization/
│   └── langchain_app.py          # Streamlit Web界面
├── requirements.txt              # Python依赖
└── README.md                     # 项目说明
```

## 技术栈

| 组件 | 技术选型 |
|------|----------|
| RAG框架 | LangChain |
| 向量数据库 | FAISS |
| 嵌入模型 | sentence-transformers/all-MiniLM-L6-v2 |
| PDF解析 | PyPDFLoader |
| 大语言模型 | Qwen / Claude |
| Web框架 | Streamlit |
| Python版本 | 3.8+ |

## 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| CHUNK_SIZE | 512 | 文本块大小 |
| CHUNK_OVERLAP | 50 | 文本块重叠长度 |
| TOP_K | 5 | 检索返回数量 |
| SIMILARITY_THRESHOLD | 0.7 | 相似度阈值 |

## 核心优势

1. **工程化**: 使用LangChain框架，快速构建RAG系统
2. **逻辑连贯性**: 递归分割保持上下文完整性
3. **高精度检索**: FAISS + 余弦相似度实现语义匹配
4. **可解释性**: 证据对齐机制标注原文来源
5. **灵活性**: 支持多模型切换，适应不同场景

## 适用场景

- 企业内部知识库问答
- 专业技术文档检索
- 学术资料问答系统
- 产品手册智能客服

## License

MIT License
