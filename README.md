# RAG 知识库问答（升级版）

基于 RAG（检索增强生成）的命令行问答程序：多文件导入 + 向量持久化 + 语义检索。

## 功能

- 支持多文件导入（docs/ 目录下的 .txt / .md）
- LangChain 文本切块（RecursiveCharacterTextSplitter）
- 智谱 embedding-3 向量化
- chromadb 持久化向量库（重启无需重新计算）
- DeepSeek 生成答案，支持"资料里没有"拒答

## 运行方法

1. 设置环境变量 `ZHIPU_KEY` 和 `DEEPSEEK_KEY`
2. 把文档放进 `docs/` 目录（支持 .txt / .md）
3. 运行 `python main.py`
4. 首次运行会向量化并存入 `chroma_db/`，之后启动直接复用

## 项目结构

```
rag-assistant/
├── main.py              主程序
├── docs/                知识库文档
├── chroma_db/           向量库（自动生成，不入库）
└── README.md
```

## 技术栈

- Python 3.12
- chromadb（向量数据库，持久化）
- 智谱 embedding-3（文本向量化，2048 维）
- LangChain text splitters（文本切块）
- DeepSeek API（答案生成）
