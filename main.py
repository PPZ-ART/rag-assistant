import os
import json
import urllib.request
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ===== 配置 =====
智谱密钥 = os.environ.get("ZHIPU_KEY")
DeepSeek密钥 = os.environ.get("DEEPSEEK_KEY")
if not 智谱密钥 or not DeepSeek密钥:
    print("错误：请先设置 ZHIPU_KEY 和 DEEPSEEK_KEY 环境变量")
    exit()

文档目录 = "docs"
向量库目录 = "chroma_db"

# ===== 1. 向量化（智谱 embedding-3）=====
def 向量化(文本):
    数据 = json.dumps({"model": "embedding-3", "input": 文本}).encode("utf-8")
    请求 = urllib.request.Request(
        "https://open.bigmodel.cn/api/paas/v4/embeddings",
        data=数据,
        headers={"Authorization": "Bearer " + 智谱密钥, "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(请求, timeout=60) as 响应:
        return json.loads(响应.read().decode("utf-8"))["data"][0]["embedding"]

# ===== 2. 持久化向量库（重启不丢）=====
客户端 = chromadb.PersistentClient(path=向量库目录)
集合 = 客户端.get_or_create_collection("knowledge")

# ===== 3. 多文件导入（LangChain 切块）=====
def 导入文档():
    切块器 = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    总块数 = 0
    for 文件名 in sorted(os.listdir(文档目录)):
        if not 文件名.endswith((".txt", ".md")):
            continue
        路径 = os.path.join(文档目录, 文件名)
        with open(路径, "r", encoding="utf-8") as f:
            内容 = f.read()
        块列表 = 切块器.split_text(内容)
        if not 块列表:
            continue
        ids = [文件名 + "#" + str(i) for i in range(len(块列表))]
        向量列表 = [向量化(块) for 块 in 块列表]
        元数据 = [{"source": 文件名} for _ in 块列表]
        集合.upsert(ids=ids, documents=块列表, embeddings=向量列表, metadatas=元数据)
        print("  " + 文件名 + ": " + str(len(块列表)) + " 块")
        总块数 += len(块列表)
    print("共导入 " + str(总块数) + " 块，库中现有 " + str(集合.count()) + " 块")

# ===== 4. 检索 =====
def 检索(问题, 取前几块=3):
    结果 = 集合.query(query_embeddings=[向量化(问题)], n_results=取前几块)
    返回块 = 结果["documents"][0] if 结果["documents"] else []
    来源 = [m.get("source", "?") for m in 结果["metadatas"][0]] if 结果["metadatas"] else []
    return 返回块, 来源

# ===== 5. 生成 =====
def 生成回答(问题, 相关块):
    上下文 = "\n\n".join(相关块)
    提示 = (
        "请根据下面的资料回答问题。如果资料中没有答案，直接说“资料里没有提到”。\n\n"
        "资料：\n" + 上下文 + "\n\n问题：" + 问题
    )
    数据 = json.dumps({
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": 提示}],
    }).encode("utf-8")
    请求 = urllib.request.Request(
        "https://api.deepseek.com/chat/completions",
        data=数据,
        headers={"Authorization": "Bearer " + DeepSeek密钥, "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(请求, timeout=60) as 响应:
        return json.loads(响应.read().decode("utf-8"))["choices"][0]["message"]["content"]

# ===== 入口 =====
if __name__ == "__main__":
    if 集合.count() == 0:
        print("向量库为空，正在导入文档...")
        导入文档()
    else:
        print("向量库已有 " + str(集合.count()) + " 块，直接使用（如需重建，删除 chroma_db 目录）")

    print("\nRAG 问答已就绪（输入“退出”结束）")
    while True:
        问题 = input("\n你问：")
        if 问题 == "退出":
            break
        相关块, 来源 = 检索(问题)
        print("\n【检索到 " + str(len(相关块)) + " 块，来源：" + "、".join(sorted(set(来源))) + "】")
        print("\n【AI 回答】")
        print(生成回答(问题, 相关块))
