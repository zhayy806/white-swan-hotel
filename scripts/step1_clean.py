#!/usr/bin/env python3
"""
OTA 酒店评论数据清洗 — 5 步流水线
==================================
输入: 影刀RPA采集的原始CSV
输出: {DATA_DIR}/cleaned_comments.csv（清洗后，待入库SQLite）

用法:
    python scripts/step1_clean.py                  # 自动查找CSV
    python scripts/step1_clean.py --csv /path/to/file.csv
    python scripts/step1_clean.py --dry-run         # 预览不保存
"""

import argparse
import os
import re
import sys
from pathlib import Path

# 将项目根目录加入 sys.path（便于从任意位置运行）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime

import config


# ============================================================
# 步骤1: 字段拆分与规范化
# ============================================================

def parse_rating(rating_raw: str) -> tuple:
    """
    输入: "4.8\n超棒" 或 "3.2" 或 "4.5 很好"
    输出: (4.8, "超棒") 或 (3.2, "")
    评分 < 4.5 通常不带文字标签
    """
    if pd.isna(rating_raw):
        return (np.nan, "")

    text = str(rating_raw).replace("\n", " ").replace("\r", " ").strip()
    # 尝试匹配 "数字 文字" 模式
    m = re.match(r"^([\d.]+)\s*(.*)", text)
    if m:
        score = float(m.group(1))
        label = m.group(2).strip()
        return (round(score, 1), label)
    return (np.nan, "")


def parse_date(date_raw: str) -> str:
    """
    输入: "于2026年7月入住" / "2026-07" / "2026年7月"
    输出: "2026-07"
    """
    if pd.isna(date_raw):
        return ""
    text = str(date_raw).strip()
    # 中文: 于2026年7月入住
    m = re.search(r"(\d{4})\s*年\s*(\d{1,2})\s*月", text)
    if m:
        return f"{m.group(1)}-{int(m.group(2)):02d}"
    # 英文: 2026-07
    m = re.search(r"(\d{4})-(\d{1,2})", text)
    if m:
        return f"{m.group(1)}-{int(m.group(2)):02d}"
    return text[:7] if len(text) >= 7 else text


def parse_user_stats(stats_raw: str) -> int:
    """
    输入: "64条点评" / "5条点评"
    输出: 64
    """
    if pd.isna(stats_raw):
        return 0
    text = str(stats_raw).strip()
    m = re.search(r"(\d+)\s*条点评", text)
    if m:
        return int(m.group(1))
    # 可能是纯数字
    try:
        return int(text)
    except ValueError:
        return 0


# ============================================================
# 步骤2: 评论文本清洗
# ============================================================

def clean_comment_text(text: str) -> str:
    """
    清洗一条评论文本:
    - 去除"展开更多"标记
    - 去除多余空白
    - 去除纯符号行
    - 统一换行为中文句号/空格
    """
    if pd.isna(text) or not str(text).strip():
        return ""

    text = str(text)

    # 去"展开更多"标记（OTA截断产物）
    text = re.sub(r"展开更多\s*$", "", text)
    text = re.sub(r"展开全部\s*$", "", text)

    # 去HTML标签残留
    text = re.sub(r"<[^>]+>", "", text)

    # 统一换行符
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 将多个连续换行压缩为一个
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 去首尾空白
    text = text.strip()

    return text


# ============================================================
# 步骤3: 去重与过滤
# ============================================================

def dedup_and_filter(df: pd.DataFrame) -> pd.DataFrame:
    """
    - 同用户+同评论文本 → 去重
    - 过短评论 → 标记 is_valid=false
    """
    initial = len(df)

    # 去重
    dup_cols = ["用户编号", "评论"]
    df = df.drop_duplicates(subset=dup_cols, keep="first")
    dup_removed = initial - len(df)
    if dup_removed:
        print(f"  🗑️  去重: 移除 {dup_removed} 条完全重复评论")

    # 标记有效性
    df["is_valid"] = df["评论"].apply(
        lambda x: len(str(x).strip()) >= config.MIN_COMMENT_LENGTH
    )
    invalid_count = (~df["is_valid"]).sum()
    if invalid_count:
        print(f"  🏷️  标记无效(短评论<{config.MIN_COMMENT_LENGTH}字): {invalid_count} 条")

    return df


# ============================================================
# 步骤4: 情感极性标记
# ============================================================

def classify_sentiment(score: float) -> str:
    if pd.isna(score):
        return "未知"
    if score >= config.SENTIMENT_POSITIVE:
        return "好评"
    elif score >= config.SENTIMENT_NEUTRAL:
        return "中评"
    else:
        return "差评"


# ============================================================
# 步骤5: 观点标签抽取（关键词正则规则匹配）
# ============================================================

def extract_dimension_tags(comment: str) -> str:
    """
    基于关键词字典匹配，返回逗号分隔的维度标签
    例如: "服务, 餐饮, 位置"
    """
    if not comment or not str(comment).strip():
        return ""

    text = str(comment)
    matched_dims = []

    for dim, keywords in config.DIMENSION_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                matched_dims.append(dim)
                break  # 该维度已命中，跳到下一个维度

    return ",".join(matched_dims)


# ============================================================
# 主流水线
# ============================================================

def load_raw_csv(csv_path: str) -> pd.DataFrame:
    """
    读取影刀RPA导出的CSV。
    由于评论文本含换行符，CSV字段被正确引号包裹，
    用 Python csv 模块逐行解析后转 DataFrame。
    """
    import csv

    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if len(row) >= 7:
                rows.append(row)

    df = pd.DataFrame(rows, columns=["评分", "用户编号", "评论", "房型", "时间", "旅游类型", "评论的评论"])
    print(f"📥 读取原始数据: {len(df)} 条")
    return df


def run_pipeline(csv_path: str) -> pd.DataFrame:
    """执行完整 5 步清洗流水线，返回清洗后 DataFrame"""
    print("=" * 60)
    print("🧹 OTA 酒店评论数据清洗流水线")
    print(f"   数据源: {csv_path}")
    print(f"   时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # --- 加载 ---
    df = load_raw_csv(csv_path)
    print(f"\n📊 原始数据: {len(df)} 行 × {len(df.columns)} 列")

    # --- 步骤1: 字段拆分 ---
    print("\n--- 步骤1: 字段拆分与规范化 ---")
    scores, labels = zip(*df["评分"].apply(parse_rating))
    df["rating_score"] = scores
    df["rating_label"] = labels
    df["review_date"] = df["时间"].apply(parse_date)
    df["user_review_count"] = df["评论的评论"].apply(parse_user_stats)

    # --- 步骤2: 文本清洗 ---
    print("\n--- 步骤2: 评论文本清洗 ---")
    df["评论"] = df["评论"].apply(clean_comment_text)
    # 清洗后可能产生空文本
    became_empty = (df["评论"].str.len() == 0).sum()
    if became_empty:
        print(f"  ⚠️  清洗后 {became_empty} 条评论变为空")

    # --- 步骤3: 去重过滤 ---
    print("\n--- 步骤3: 去重与过滤 ---")
    df = dedup_and_filter(df)

    # --- 步骤4: 情感标记 ---
    print("\n--- 步骤4: 情感极性标记 ---")
    df["sentiment"] = df["rating_score"].apply(classify_sentiment)
    sentiment_counts = df["sentiment"].value_counts()
    for lbl, cnt in sentiment_counts.items():
        print(f"  {lbl}: {cnt} 条")

    # --- 步骤5: 观点标签 ---
    print("\n--- 步骤5: 观点标签抽取 ---")
    df["dimension_tags"] = df["评论"].apply(extract_dimension_tags)
    tagged = (df["dimension_tags"] != "").sum()
    print(f"  命中观点标签: {tagged} 条 ({tagged/len(df)*100:.1f}%)")

    # --- 精简输出列 ---
    output_cols = [
        "用户编号", "评论", "房型", "旅游类型",
        "rating_score", "rating_label", "review_date",
        "sentiment", "dimension_tags", "user_review_count",
        "is_valid",
    ]
    df_out = df[output_cols].copy()
    df_out.insert(0, "comment_id", range(1, len(df_out) + 1))
    df_out["hotel_name"] = "白天鹅宾馆"  # 当前只采集了这一家

    print(f"\n✅ 清洗完成: {len(df_out)} 条有效数据")
    print(f"   输出字段: {list(df_out.columns)}")
    return df_out


def main():
    parser = argparse.ArgumentParser(description="OTA酒店评论数据清洗")
    parser.add_argument("--csv", type=str, help="原始CSV路径")
    parser.add_argument("--output", type=str, help="输出路径（默认 data/cleaned_comments.csv）")
    parser.add_argument("--dry-run", action="store_true", help="预览模式，不保存文件")
    args = parser.parse_args()

    # 确定CSV路径
    csv_path = args.csv or config.CSV_SOURCE
    if not os.path.exists(csv_path):
        csv_path = config.CSV_FALLBACK
    if not os.path.exists(csv_path):
        print(f"❌ 找不到CSV文件: {csv_path}")
        print("   请用 --csv 指定路径，或把文件放到 data/ 目录")
        sys.exit(1)

    # 运行清洗
    df_clean = run_pipeline(csv_path)

    # 保存
    if not args.dry_run:
        output_path = args.output or os.path.join(config.DATA_DIR, "cleaned_comments.csv")
        df_clean.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"\n💾 清洗结果已保存: {output_path}")
        print(f"   文件大小: {os.path.getsize(output_path)/1024:.1f} KB")

        # 统计摘要
        print(f"\n📋 数据统计摘要:")
        print(f"   有效评论: {df_clean['is_valid'].sum()} 条")
        print(f"   无效评论: {(~df_clean['is_valid']).sum()} 条")
        print(f"   平均评分: {df_clean['rating_score'].mean():.2f}")
        print(f"   日期范围: {df_clean['review_date'].min()} ~ {df_clean['review_date'].max()}")
        print(f"   房型数: {df_clean['房型'].nunique()}")
        print(f"   覆盖维度: {sum(df_clean['dimension_tags'] != '')} 条")
    else:
        print("\n🔍 [dry-run] 预览模式，未保存文件")
        print(df_clean.head(10).to_string())


if __name__ == "__main__":
    main()
