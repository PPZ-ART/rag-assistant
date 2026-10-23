import os
import json
import math
import urllib.request

# ===== 配置：从环境变量读 Key =====
智谱密钥 = os.environ.get("ZHIPU_KEY")
DeepSeek密钥 = os.environ.get("DEEPSEEK_KEY")
if not 智谱密钥 or not DeepSeek密钥:
    print("错误：请先设置 ZHIPU_KEY 和 DEEPSEEK_KEY 环境变量")
    exit()

# ===== 1. 读文档并切块 =====
def 读文档并切块(路径, 块大小=300):
    with open(路径, "r", encoding="utf-8") as f:
        全文 = f.read()
    块列表 = []
    当前块 = ""
    for 段落 in 全文.split("\n"):
        段落 = 段落.strip()
        if not 段落:
            continue
        if len(当前块) + len(段落) > 块大小 and 当前块:
            块列表.append(当前块)
            当前块 = 段落
        else:
            当前块 = 当前块 + "\n" + 段落 if 当前块 else 段落
    if 当前块:
        块列表.append(当前块)
    return 块列表

# ===== 2. 调智谱 API 把文字变成向量 =====
def 向量化(文本):
    数据 = json.dumps({"model": "embedding-3", "input": 文本}).encode("utf-8")
    请求 = urllib.request.Request(
        "https://open.bigmodel.cn/api/paas/v4/embeddings",
        data=数据,
        headers={"Authorization": "Bearer " + 智谱密钥, "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(请求, timeout=60) as 响应:
        结果 = json.loads(响应.read().decode("utf-8"))
    return 结果["data"][0]["embedding"]

# ===== 3. 余弦相似度：算两个向量有多像 =====
def 余弦相似度(向量a, 向量b):
    点积 = sum(x * y for x, y in zip(向量a, 向量b))
    模a = math.sqrt(sum(x * x for x in 向量a))
    模b = math.sqrt(sum(x * x for x in 向量b))
    return 点积 / (模a * 模b)

# ===== 4. 检索：找出和问题最相关的几块 =====
def 检索(问题, 块列表, 块向量列表, 取前几块=3):
    问题向量 = 向量化(问题)
    得分列表 = [(余弦相似度(问题向量, 块向量), i) for i, 块向量 in enumerate(块向量列表)]
    得分列表.sort(reverse=True)
    return [块列表[i] for _, i in 得分列表[:取前几块]]

# ===== 5. 生成：把相关块和问题一起交给 DeepSeek =====
def 生成回答(问题, 相关块):
    上下文 = "\n\n".join(相关块)
    提示 = (
        "请根据下面的资料回答问题。如果资料中没有答案，就直接说“资料里没有提到”。\n\n"
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
        结果 = json.loads(响应.read().decode("utf-8"))
    return 结果["choices"][0]["message"]["content"]

# ===== 主流程 =====
if __name__ == "__main__":
    文档路径 = "知识库.txt"
    print("正在读取并切块文档...")
    块列表 = 读文档并切块(文档路径)
    print("共切出", len(块列表), "个文本块")

    print("正在向量化（第一次会稍慢）...")
    块向量列表 = [向量化(块) for 块 in 块列表]
    print("向量化完成")

    print("\nRAG 问答已就绪（输入“退出”结束）")
    while True:
        问题 = input("\n你问：")
        if 问题 == "退出":
            break
        相关块 = 检索(问题, 块列表, 块向量列表)
        print("\n【检索到的最相关资料】")
        for i, 块 in enumerate(相关块):
            print("  [", i + 1, "]", 块[:50].replace("\n", " "), "...")
        print("\n【AI 回答】")
        print(生成回答(问题, 相关块))
