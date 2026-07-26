#!/usr/bin/env python3
"""
RAG 知识库构建 — SQLite → Chroma 向量化入库
=============================================
读取 SQLite 有效评论 → 文本分块 → 中文 Embedding → Chroma 持久化

用法:
    python scripts/step3_build_rag.py              # 全量构建
    python scripts/step3_build_rag.py --resume     # 增量追加（跳过已有）
    python scripts/step3_build_rag.py --test       # 构建后跑一条测试问答
"""

import argparse
import os
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

import config


# ============================================================
# 文本分块器
# ============================================================

def build_splitter() -> RecursiveCharacterTextSplitter:
    """中文友好的文本分块器"""
    return RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        length_function=len,
    )


# ============================================================
# SQLite → Documents
# ============================================================

def load_valid_comments(db_path: str) -> list[dict]:
    """读取有效评论（带丰富元数据）"""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT c.*, h.hotel_name, h.city
        FROM comment c
        JOIN hotel h ON c.hotel_id = h.hotel_id
        WHERE c.is_valid = 1
        ORDER BY c.comment_id
    """).fetchall()
    conn.close()

    comments = [dict(r) for r in rows]
    print(f"📖 从 SQLite 读取 {len(comments)} 条有效评论")
    return comments


def comments_to_documents(comments: list[dict]) -> list[Document]:
    """
    每条评论 → 1 个 LangChain Document。
    评论普遍 50-500 字，不做分块，
    超长评论（>CHUNK_SIZE）才切割。
    """
    splitter = build_splitter()
    docs = []
    skipped_long = 0

    for c in comments:
        text = c["comment_text"]
        if not text or len(text.strip()) < config.MIN_COMMENT_LENGTH:
            continue

        metadata = {
            "comment_id": c["comment_id"],
            "hotel_name": c.get("hotel_name", ""),
            "city": c.get("city", ""),
            "room_type": c.get("room_type", ""),
            "travel_type": c.get("travel_type", ""),
            "rating_score": c.get("rating_score", 0),
            "rating_label": c.get("rating_label", ""),
            "review_date": c.get("review_date", ""),
            "sentiment": c.get("sentiment", ""),
            "dimension_tags": c.get("dimension_tags", ""),
            "source": "user_review",
        }

        if len(text) <= config.CHUNK_SIZE:
            docs.append(Document(page_content=text, metadata=metadata))
        else:
            chunks = splitter.split_text(text)
            for i, chunk in enumerate(chunks):
                m = metadata.copy()
                m["chunk_index"] = i
                docs.append(Document(page_content=chunk, metadata=m))
            skipped_long += 1

    if skipped_long:
        print(f"  ✂️  {skipped_long} 条长评论被分块")
    print(f"📄 生成 {len(docs)} 个文档片段")
    return docs


# ============================================================
# Embedding 模型
# ============================================================

def build_embedding():
    """加载本地中文 Embedding 模型（首次运行自动下载）"""
    print(f"🔧 加载 Embedding 模型: {config.EMBEDDING_MODEL}")
    return HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


# ============================================================
# Chroma 向量库
# ============================================================

def build_vectorstore(docs: list[Document], resume: bool = False):
    """
    将文档向量化存入 Chroma。
    resume=True 时只追加新文档，不重建。
    """
    persist_dir = config.CHROMA_DIR
    embedding = build_embedding()

    if resume and os.path.exists(persist_dir) and os.listdir(persist_dir):
        print(f"📂 加载已有向量库: {persist_dir}")
        vectorstore = Chroma(
            persist_directory=persist_dir,
            embedding_function=embedding,
        )
        existing_count = vectorstore._collection.count()
        print(f"   已有 {existing_count} 条向量")

        # 过滤已存在的 comment_id，只追加新增
        existing_ids = set()
        # Chroma metadatas 返回列表
        batch_size = 1000
        offset = 0
        while True:
            batch = vectorstore._collection.get(
                offset=offset, limit=batch_size,
                include=["metadatas"]
            )
            if not batch["ids"]:
                break
            for meta in batch["metadatas"]:
                if meta and "comment_id" in meta:
                    existing_ids.add(meta["comment_id"])
            offset += batch_size

        new_docs = [d for d in docs if d.metadata.get("comment_id") not in existing_ids]
        skipped = len(docs) - len(new_docs)
        if skipped:
            print(f"   跳过已存在: {skipped} 条")
        if not new_docs:
            print("✅ 所有评论已向量化，无需更新")
            return vectorstore

        print(f"   新增: {len(new_docs)} 条 → 追加中...")
        # 分批追加
        batch_add(vectorstore, new_docs)
    else:
        if os.path.exists(persist_dir) and os.listdir(persist_dir):
            print(f"🗑️  清空旧向量库: {persist_dir}")
            import shutil
            shutil.rmtree(persist_dir)
            os.makedirs(persist_dir)

        print(f"🆕 新建向量库: {persist_dir}")
        vectorstore = Chroma(
            persist_directory=persist_dir,
            embedding_function=embedding,
        )
        batch_add(vectorstore, docs)

    return vectorstore


def batch_add(vectorstore, docs: list[Document], batch_size: int = 200):
    """分批添加文档到 Chroma（避免 OOM，做进度显示）"""
    total = len(docs)
    for i in range(0, total, batch_size):
        batch = docs[i:i + batch_size]
        vectorstore.add_documents(batch)
        done = min(i + batch_size, total)
        print(f"  ⏳ {done}/{total} ({done*100//total}%)", end="\r")

    print(f"\n✅ 全部入库: {total} 条向量")


# ============================================================
# 测试问答
# ============================================================

def test_rag():
    """用 RAG 链跑一条测试问答，验证向量库工作正常"""
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser

    PROMPT = """你是OTA酒店评论分析助手。请严格依据以下真实住客评论回答问题。

检索到的相关评论：
---
{context}
---

用户问题：{question}

要求：
1. 基于评论内容作答，引用具体评论中的细节
2. 如果评论中没有相关信息，明确说明"当前评论未提及"
3. 结尾标注引用的评论日期和房型
4. 客观呈现好评和差评两方面的观点

回答："""

    persist_dir = config.CHROMA_DIR
    if not os.path.exists(persist_dir) or not os.listdir(persist_dir):
        print("❌ 向量库不存在，请先运行构建")
        return

    embedding = build_embedding()
    vectorstore = Chroma(
        persist_directory=persist_dir,
        embedding_function=embedding,
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": config.RETRIEVER_K})

    # 优先用 DeepSeek，没有 key 则尝试 OpenAI
    api_key = config.LLM_API_KEY
    if not api_key:
        print("\n⚠️  未设置 LLM API Key，跳过问答测试")
        print("   设置方式: export DEEPSEEK_API_KEY='your-key'")
        print("   测试仅验证检索结果:\n")

        # 只展示检索结果
        test_queries = [
            "白天鹅宾馆的服务态度怎么样？",
            "带小孩入住推荐哪个房型？",
            "差评都在吐槽什么问题？",
        ]
        for q in test_queries:
            docs = retriever.get_relevant_documents(q)
            print(f"🔍 {q}")
            for i, doc in enumerate(docs, 1):
                meta = doc.metadata
                print(f"   [{i}] {meta.get('review_date','')} | {meta.get('room_type','')} "
                      f"| 评分{meta.get('rating_score','')} | {meta.get('sentiment','')}")
                print(f"       {doc.page_content[:120]}...")
            print()
        return

    llm = ChatOpenAI(
        base_url=config.LLM_BASE_URL,
        api_key=api_key,
        model=config.LLM_MODEL,
        temperature=0.3,
        max_tokens=1024,
    )

    prompt = ChatPromptTemplate.from_template(PROMPT)

    def format_docs(docs):
        parts = []
        for i, doc in enumerate(docs, 1):
            meta = doc.metadata
            parts.append(
                f"[评论{i}] 日期:{meta.get('review_date','?')} "
                f"房型:{meta.get('room_type','?')} "
                f"评分:{meta.get('rating_score','?')} "
                f"维度:{meta.get('dimension_tags','')}\n"
                f"{doc.page_content}"
            )
        return "\n\n".join(parts)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt | llm | StrOutputParser()
    )

    test_questions = [
        "白天鹅宾馆的服务态度怎么样？",
        "带小孩的亲子家庭入住有什么推荐和注意事项？",
        "差评主要吐槽哪些问题？",
    ]

    for q in test_questions:
        print(f"\n{'='*60}")
        print(f"🤔 {q}")
        print(f"{'='*60}")
        answer = chain.invoke(q)
        print(answer)


# ============================================================
# 主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="RAG 向量知识库构建")
    parser.add_argument("--resume", action="store_true", help="增量追加模式")
    parser.add_argument("--test", action="store_true", help="构建后跑测试问答")
    args = parser.parse_args()

    print("=" * 60)
    print("🧠 RAG 评论知识库构建")
    print(f"   数据库: {config.SQLITE_DB}")
    print(f"   向量库: {config.CHROMA_DIR}")
    print("=" * 60)

    # 1. 读 SQLite
    comments = load_valid_comments(config.SQLITE_DB)

    # 2. 转 Document
    docs = comments_to_documents(comments)

    # 3. 向量化入库
    build_vectorstore(docs, resume=args.resume)

    # 4. 输出元数据字段（供后续过滤查询参考）
    print(f"\n📋 可过滤元数据字段:")
    print(f"   sentiment: 好评 / 中评 / 差评")
    print(f"   room_type: 9 种房型")
    print(f"   travel_type: 7 种出行类型")
    print(f"   dimension_tags: 服务,卫生,设施,餐饮,位置,性价比,环境")
    print(f"   review_date: {comments[0]['review_date']} ~ {comments[-1]['review_date']}")

    # 向量库大小
    persist_dir = config.CHROMA_DIR
    if os.path.exists(persist_dir):
        total_size = 0
        for root, dirs, files in os.walk(persist_dir):
            for f in files:
                total_size += os.path.getsize(os.path.join(root, f))
        print(f"\n💾 向量库大小: {total_size/1024/1024:.1f} MB")

    # 测试
    if args.test:
        test_rag()


if __name__ == "__main__":
    main()
