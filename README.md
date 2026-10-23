# RAG 知识库问答

一个基于 RAG（检索增强生成）的命令行问答程序：让 AI 基于你自己的文档回答问题。

## 工作原理

1. 读取文档并切块
2. 用智谱 embedding-3 把每块变成向量
3. 提问时，把问题也变成向量，用余弦相似度找出最相关的几块
4. 把相关块 + 问题一起交给 DeepSeek 生成答案

## 运行方法

1. 设置环境变量 `ZHIPU_KEY`（智谱 API Key）
2. 设置环境变量 `DEEPSEEK_KEY`（DeepSeek API Key）
3. 复制 `知识库示例.txt` 为 `知识库.txt`（或把自己的文档保存为 `知识库.txt`）
4. 运行 `python main.py`

> `知识库.txt` 已在 `.gitignore` 中，不会上传到 GitHub。

## 技术栈

- Python 3.12
- 智谱 embedding-3（文本向量化）
- DeepSeek API（答案生成）
- urllib / json / math
