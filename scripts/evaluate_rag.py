#!/usr/bin/env python3
"""
RAG 检索质量评估脚本
====================
评估指标:
  - Precision@K : 检索出的 K 条中，真正相关的比例（准不准）
  - Recall@K    : 所有相关文档中，检索出 K 条能覆盖多少（全不全）
  - MRR         : 第一个相关文档排在第几位（排得好不好）
  - NDCG@K      : 考虑排序位置的相关性得分

测试方法:
  利用 review 自带的 dimension_tags 做 ground truth。
  例如: 查询"服务怎么样"，期望召回 dimension_tags 含"服务"的评论。
  这样 4616 条评论就是天然标注数据集，不需要人工标注。

用法:
    python scripts/evaluate_rag.py
    python scripts/evaluate_rag.py --k 5
"""

import argparse
import os
import sys
import time
import json
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

import config

# ============================================================
# 测试查询集 — 每个维度构造自然语言问题
# ============================================================

TEST_QUERIES = {
    "服务": [
        "服务态度怎么样",
        "前台办理入住快不快",
        "工作人员热情吗",
    ],
    "卫生": [
        "房间干净吗",
        "卫生条件好不好",
        "床品毛巾干净程度",
    ],
    "设施": [
        "酒店设施新旧程度",
        "房间隔音效果如何",
        "空调淋浴设备怎么样",
    ],
    "餐饮": [
        "早餐好吃吗",
        "酒店餐厅口味如何",
        "行政酒廊的下午茶怎么样",
    ],
    "位置": [
        "酒店位置方便吗",
        "周边交通便利程度",
        "离地铁站远不远",
    ],
    "性价比": [
        "这个酒店值不值这个价",
        "性价比高吗",
    ],
    "环境": [
        "江景好看吗",
        "泳池和花园环境怎么样",
        "大堂装修风格如何",
    ],
}

# 综合查询（跨维度）
COMPREHENSIVE_QUERIES = [
    ("带小孩入住推荐什么房型", ["环境", "服务", "设施"]),
    ("差评都在吐槽什么问题", []),  # 特殊：找差评
    ("白天鹅最大的优点是什么", []),
    ("情侣出游体验怎么样", []),
    ("商务出差住这里合适吗", []),
]


# ============================================================
# 判断文档是否相关
# ============================================================

def is_relevant(doc_metadata: dict, target_dimension: str) -> bool:
    """检查文档的 dimension_tags 是否包含目标维度"""
    tags = doc_metadata.get("dimension_tags", "")
    return target_dimension in tags.split(",")


def is_negative(doc_metadata: dict) -> bool:
    """检查是否为差评"""
    return doc_metadata.get("sentiment") == "差评"


# ============================================================
# 单次检索评估
# ============================================================

def evaluate_single_query(
    vectorstore,
    query: str,
    relevance_fn,
    k: int = 10,
) -> dict:
    """
    对单次查询评估，返回多项指标。

    relevance_fn(metadata) → bool: 判断文档是否相关
    """
    results = vectorstore.similarity_search_with_relevance_scores(query, k=k)

    # 相关性标签: 1=相关, 0=不相关
    relevance = [1 if relevance_fn(doc.metadata) else 0 for doc, score in results]
    scores = [score for doc, score in results]

    # ----- Precision@K -----
    # 检索出的K条中，相关的占多少
    precision_at_k = {}
    for kval in [1, 3, 5, k]:
        if kval <= len(relevance):
            precision_at_k[kval] = sum(relevance[:kval]) / kval

    # ----- Recall@K -----
    # 注意: 这里用向量库中 total relevant 做分母（近似）
    # 用元数据过滤出所有相关文档数作为 ground truth
    total_relevant = vectorstore._collection.count()  # fallback
    recall_at_k = {}
    for kval in [1, 3, 5, k]:
        if kval <= len(relevance):
            # 由于无法高效获取全部相关文档数，这里用近似:
            # 用同维度查询 + metadata filter 获取总数
            recall_at_k[kval] = sum(relevance[:kval])  # 返回绝对命中数

    # ----- MRR (Mean Reciprocal Rank) -----
    # 第一个相关文档的排名倒数
    mrr = 0.0
    for i, rel in enumerate(relevance, 1):
        if rel == 1:
            mrr = 1.0 / i
            break

    # ----- NDCG@K -----
    # 考虑排序位置，越靠前权重越高
    def dcg_at_k(rel_list):
        dcg = 0.0
        for i, rel in enumerate(rel_list, 1):
            dcg += rel / np.log2(i + 1)
        return dcg

    def ndcg_at_k(rel_list):
        dcg = dcg_at_k(rel_list)
        ideal = dcg_at_k(sorted(rel_list, reverse=True))
        return dcg / ideal if ideal > 0 else 0.0

    ndcg = {}
    for kval in [1, 3, 5, k]:
        if kval <= len(relevance):
            ndcg[kval] = ndcg_at_k(relevance[:kval])

    return {
        "query": query,
        "precision": precision_at_k,
        "recall_hits": recall_at_k,
        "mrr": mrr,
        "ndcg": ndcg,
        "relevance": relevance,
        "top_scores": scores[:5],
    }


# ============================================================
# 全量评估
# ============================================================

def run_full_evaluation(k: int = 10):
    """遍历所有测试查询，汇总评估指标"""
    persist_dir = config.CHROMA_DIR
    if not os.path.exists(persist_dir):
        print("❌ 向量库不存在，请先运行 step3_build_rag.py")
        return

    embedding = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = Chroma(
        persist_directory=persist_dir,
        embedding_function=embedding,
    )

    total_docs = vectorstore._collection.count()
    print(f"📊 向量库文档总数: {total_docs}")
    print(f"   Top-K 设置: {k}")
    print(f"   Embedding 模型: {config.EMBEDDING_MODEL}")
    print(f"   测试查询数: {sum(len(v) for v in TEST_QUERIES.values())} 维度查询 "
          f"+ {len(COMPREHENSIVE_QUERIES)} 综合查询")
    print()

    all_results = []

    # ---- 维度查询评估 ----
    print("=" * 70)
    print("📋 维度标签查询评估")
    print("=" * 70)

    dimension_metrics = defaultdict(list)

    for dim, queries in TEST_QUERIES.items():
        for q in queries:
            result = evaluate_single_query(
                vectorstore, q,
                relevance_fn=lambda m: is_relevant(m, dim),
                k=k,
            )
            all_results.append(result)
            dimension_metrics[dim].append(result)

            # 实时打印
            p3 = result["precision"].get(3, 0)
            p5 = result["precision"].get(5, 0)
            print(f"  {dim:6s} | {q:22s} | P@3={p3:.2f} P@5={p5:.2f} "
                  f"MRR={result['mrr']:.2f} | hits={result['recall_hits'].get(3,0)}")

    # ---- 综合查询评估 ----
    print("\n" + "=" * 70)
    print("📋 综合查询评估（无预设维度标签，展示检索内容）")
    print("=" * 70)

    for q, expected_dims in COMPREHENSIVE_QUERIES:
        docs = vectorstore.similarity_search(q, k=3)
        print(f"\n  🔍 {q}")
        for i, doc in enumerate(docs, 1):
            m = doc.metadata
            print(f"     [{i}] {m.get('review_date','')} | {m.get('room_type','')} "
                  f"| 评分{m.get('rating_score','')} | {m.get('sentiment','')}")
            dims = m.get("dimension_tags", "")
            if expected_dims:
                match = any(d in dims for d in expected_dims)
                status = "✅" if match else "❓"
                print(f"         {status} 维度:{dims} | {doc.page_content[:80]}...")
            else:
                print(f"         维度:{dims} | {doc.page_content[:80]}...")

    # ---- 汇总统计 ----
    print("\n" + "=" * 70)
    print("📊 检索质量汇总报告")
    print("=" * 70)

    precisions = defaultdict(list)
    mrrs = []
    ndcgs = defaultdict(list)

    for r in all_results:
        for kv, pv in r["precision"].items():
            precisions[kv].append(pv)
        mrrs.append(r["mrr"])
        for kv, nv in r["ndcg"].items():
            ndcgs[kv].append(nv)

    print(f"\n{'指标':<20} {'均值':>8} {'中位数':>8} {'最低':>8} {'最高':>8}")
    print("-" * 60)

    for kval in sorted(precisions.keys()):
        vals = precisions[kval]
        print(f"{'Precision@'+str(kval):<20} {np.mean(vals):>8.4f} {np.median(vals):>8.4f} "
              f"{np.min(vals):>8.4f} {np.max(vals):>8.4f}")

    print(f"{'MRR':<20} {np.mean(mrrs):>8.4f} {np.median(mrrs):>8.4f} "
          f"{np.min(mrrs):>8.4f} {np.max(mrrs):>8.4f}")

    for kval in sorted(ndcgs.keys()):
        vals = ndcgs[kval]
        print(f"{'NDCG@'+str(kval):<20} {np.mean(vals):>8.4f} {np.median(vals):>8.4f} "
              f"{np.min(vals):>8.4f} {np.max(vals):>8.4f}")

    # ---- 按维度分组 ----
    print(f"\n{'─'*60}")
    print(f"{'维度':<8} {'查询数':>6} {'P@3均值':>8} {'P@5均值':>8} {'MRR均值':>8}")
    print(f"{'─'*60}")
    for dim in sorted(dimension_metrics.keys()):
        vals = dimension_metrics[dim]
        p3 = np.mean([v["precision"].get(3, 0) for v in vals])
        p5 = np.mean([v["precision"].get(5, 0) for v in vals])
        mrr = np.mean([v["mrr"] for v in vals])
        print(f"{dim:<8} {len(vals):>6} {p3:>8.4f} {p5:>8.4f} {mrr:>8.4f}")

    # ---- 失败查询分析 ----
    print(f"\n{'─'*60}")
    print("⚠️  低质量查询 (MRR=0 即第一个相关文档未出现在Top-K):")
    print(f"{'─'*60}")
    failures = [r for r in all_results if r["mrr"] == 0]
    if failures:
        for r in failures:
            print(f"  ❌ {r['query']}")
            print(f"     检索到的维度: {[doc.metadata.get('dimension_tags','')[:30] for doc, _ in zip([r]*len(r['relevance']), r['relevance'])] if False else ''}")
    else:
        print("  🎉 全部查询都命中了相关文档！")

    # ---- 总结 ----
    avg_p3 = np.mean(precisions[3])
    avg_p5 = np.mean(precisions[5])
    avg_mrr = np.mean(mrrs)

    print(f"\n{'='*70}")
    print(f"🏆 综合评分")
    print(f"{'='*70}")
    print(f"  Precision@3 = {avg_p3:.2%}  (检索3条中相关的占比)")
    print(f"  Precision@5 = {avg_p5:.2%}  (检索5条中相关的占比)")
    print(f"  MRR         = {avg_mrr:.2%}  (第一个相关结果的平均排名倒数)")
    print(f"  Top-K       = {k}")
    print(f"  测试查询数   = {len(all_results)}")

    # 评级
    if avg_p3 >= 0.9:
        grade = "A+ (优秀)"
    elif avg_p3 >= 0.8:
        grade = "A (良好)"
    elif avg_p3 >= 0.7:
        grade = "B (中等)"
    elif avg_p3 >= 0.6:
        grade = "C (及格)"
    else:
        grade = "D (需优化)"
    print(f"  综合评级     = {grade}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="RAG 检索质量评估")
    parser.add_argument("--k", type=int, default=10, help="Top-K (默认10)")
    parser.add_argument("--json", type=str, help="导出详细结果到JSON文件")
    args = parser.parse_args()

    start = time.time()
    results = run_full_evaluation(k=args.k)
    elapsed = time.time() - start
    print(f"\n⏱️  评估耗时: {elapsed:.1f}s")

    if args.json and results:
        with open(args.json, "w") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"💾 详细结果已导出: {args.json}")


if __name__ == "__main__":
    main()
