#!/usr/bin/env python3
"""
SQLite 入库脚本 — 建表 + 导入清洗后的数据
==========================================
输入: data/cleaned_comments.csv（step1_clean.py 产出）
输出: db/ai_data_agent.db

用法:
    python scripts/step2_load_db.py
    python scripts/step2_load_db.py --csv data/cleaned_comments.csv
    python scripts/step2_load_db.py --stats  # 仅打印统计，不入库
"""

import argparse
import os
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import config


# ============================================================
# 建表 DDL
# ============================================================

DDL_STATEMENTS = """
-- 酒店基础信息表
CREATE TABLE IF NOT EXISTS hotel (
    hotel_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    hotel_name  TEXT    NOT NULL,           -- 酒店名称
    city        TEXT    DEFAULT '广州',      -- 城市
    district    TEXT    DEFAULT '荔湾区',     -- 区域
    star_rating INTEGER DEFAULT 5,           -- 星级
    address     TEXT,                        -- 地址
    created_at  TEXT    DEFAULT (datetime('now','localtime'))
);

-- 评论主表
CREATE TABLE IF NOT EXISTS comment (
    comment_id        INTEGER PRIMARY KEY,
    hotel_id          INTEGER NOT NULL DEFAULT 1,  -- 关联酒店
    user_id           TEXT,                        -- 用户编号(脱敏)
    comment_text      TEXT    NOT NULL,             -- 评论正文
    room_type         TEXT,                        -- 房型
    travel_type       TEXT,                        -- 旅游类型
    rating_score      REAL,                        -- 评分(0-5)
    rating_label      TEXT,                        -- 评分标签(超棒/很好)
    review_date       TEXT,                        -- 入住日期 YYYY-MM
    sentiment         TEXT,                        -- 情感: 好评/中评/差评
    dimension_tags    TEXT,                        -- 观点标签(逗号分隔)
    user_review_count INTEGER DEFAULT 0,           -- 用户历史点评数
    is_valid          INTEGER DEFAULT 1,           -- 是否有效(1=有效)
    created_at        TEXT DEFAULT (datetime('now','localtime')),
    FOREIGN KEY (hotel_id) REFERENCES hotel(hotel_id)
);

-- RPA采集日志表（供前端看板展示采集状态）
CREATE TABLE IF NOT EXISTS crawl_log (
    log_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_no    TEXT    NOT NULL,           -- 批次号
    hotel_name  TEXT,                       -- 采集酒店
    total_count INTEGER DEFAULT 0,          -- 本次抓取条数
    success_count INTEGER DEFAULT 0,        -- 成功条数
    fail_count  INTEGER DEFAULT 0,          -- 失败条数
    fail_detail TEXT,                       -- 失败详情(JSON)
    csv_file    TEXT,                       -- 源CSV文件名
    started_at  TEXT,                       -- 开始时间
    finished_at TEXT,                       -- 结束时间
    created_at  TEXT DEFAULT (datetime('now','localtime'))
);

-- 观点标签字典表
CREATE TABLE IF NOT EXISTS dimension_tag (
    tag_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    tag_name TEXT NOT NULL UNIQUE,          -- 标签名: 服务/卫生/设施/餐饮/位置/性价比/环境
    keywords TEXT                           -- 关键词(JSON数组)
);

-- 索引
CREATE INDEX IF NOT EXISTS idx_comment_hotel     ON comment(hotel_id);
CREATE INDEX IF NOT EXISTS idx_comment_sentiment ON comment(sentiment);
CREATE INDEX IF NOT EXISTS idx_comment_date      ON comment(review_date);
CREATE INDEX IF NOT EXISTS idx_comment_rating    ON comment(rating_score);
CREATE INDEX IF NOT EXISTS idx_comment_room      ON comment(room_type);
CREATE INDEX IF NOT EXISTS idx_comment_valid     ON comment(is_valid);
"""


def create_tables(db_path: str):
    """建表 + 初始化字典数据"""
    conn = sqlite3.connect(db_path)
    conn.executescript(DDL_STATEMENTS)

    # 插入默认酒店记录
    conn.execute("""
        INSERT OR IGNORE INTO hotel (hotel_id, hotel_name, city, district, star_rating, address)
        VALUES (1, '白天鹅宾馆', '广州', '荔湾区', 5, '广州市荔湾区沙面南街1号')
    """)

    # 插入观点标签字典
    import json
    for dim, kws in config.DIMENSION_KEYWORDS.items():
        conn.execute("""
            INSERT OR REPLACE INTO dimension_tag (tag_name, keywords)
            VALUES (?, ?)
        """, (dim, json.dumps(kws, ensure_ascii=False)))

    conn.commit()
    conn.close()
    print("✅ SQLite 表结构创建完成")


def load_cleaned_csv(csv_path: str, db_path: str):
    """将清洗后的CSV导入 comment 表"""
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    print(f"📥 读取清洗数据: {len(df)} 条")

    conn = sqlite3.connect(db_path)

    # 清空旧评论（保留 hotel 和字典）
    conn.execute("DELETE FROM comment")

    for _, row in df.iterrows():
        conn.execute("""
            INSERT INTO comment (
                comment_id, hotel_id, user_id, comment_text,
                room_type, travel_type, rating_score, rating_label,
                review_date, sentiment, dimension_tags,
                user_review_count, is_valid
            ) VALUES (?, 1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            int(row["comment_id"]),
            str(row.get("用户编号", "")),
            str(row.get("评论", "")),
            str(row.get("房型", "")),
            str(row.get("旅游类型", "")),
            float(row["rating_score"]) if pd.notna(row.get("rating_score")) else None,
            str(row.get("rating_label", "")),
            str(row.get("review_date", "")),
            str(row.get("sentiment", "")),
            str(row.get("dimension_tags", "")),
            int(row.get("user_review_count", 0)),
            int(row.get("is_valid", 1)),
        ))

    conn.commit()
    conn.close()
    print(f"✅ {len(df)} 条评论已写入 SQLite")


def record_crawl_log(csv_source: str, db_path: str):
    """记录本次采集批次日志"""
    import glob
    csv_name = os.path.basename(csv_source)
    conn = sqlite3.connect(db_path)

    total = conn.execute("SELECT COUNT(*) FROM comment").fetchone()[0]
    valid = conn.execute("SELECT COUNT(*) FROM comment WHERE is_valid=1").fetchone()[0]
    invalid = total - valid

    now = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    batch_no = f"BATCH-{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}"

    conn.execute("""
        INSERT INTO crawl_log (batch_no, hotel_name, total_count, success_count,
                               fail_count, fail_detail, csv_file, started_at, finished_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (batch_no, "白天鹅宾馆", total, valid, invalid,
          f"短评论<{config.MIN_COMMENT_LENGTH}字: {invalid}条",
          csv_name, now, now))
    conn.commit()
    conn.close()
    print(f"📝 采集日志已记录: {batch_no}")


def print_db_stats(db_path: str):
    """打印数据库统计摘要"""
    conn = sqlite3.connect(db_path)

    print("\n" + "=" * 60)
    print("📊 SQLite 数据统计")
    print("=" * 60)

    # 总量
    total = conn.execute("SELECT COUNT(*) FROM comment").fetchone()[0]
    valid = conn.execute("SELECT COUNT(*) FROM comment WHERE is_valid=1").fetchone()[0]
    print(f"\n总评论: {total}  |  有效: {valid}  |  无效: {total - valid}")

    # 情感分布
    print("\n--- 情感分布 ---")
    for row in conn.execute("""
        SELECT sentiment, COUNT(*) as cnt, ROUND(AVG(rating_score),2) as avg_score
        FROM comment WHERE is_valid=1
        GROUP BY sentiment ORDER BY avg_score DESC
    """):
        print(f"  {row[0]}: {row[1]} 条 (均分 {row[2]})")

    # 房型分布
    print("\n--- 房型 TOP5 ---")
    for row in conn.execute("""
        SELECT room_type, COUNT(*) as cnt, ROUND(AVG(rating_score),2) as avg
        FROM comment WHERE is_valid=1
        GROUP BY room_type ORDER BY cnt DESC LIMIT 5
    """):
        print(f"  {row[0]}: {row[1]} 条 (均分 {row[2]})")

    # 旅游类型
    print("\n--- 旅游类型分布 ---")
    for row in conn.execute("""
        SELECT travel_type, COUNT(*) as cnt FROM comment WHERE is_valid=1
        GROUP BY travel_type ORDER BY cnt DESC
    """):
        print(f"  {row[0]}: {row[1]} 条")

    # 时间范围
    date_range = conn.execute("""
        SELECT MIN(review_date), MAX(review_date) FROM comment WHERE is_valid=1
    """).fetchone()
    print(f"\n时间范围: {date_range[0]} ~ {date_range[1]}")

    # 维度标签覆盖
    tagged = conn.execute("""
        SELECT COUNT(*) FROM comment WHERE is_valid=1 AND dimension_tags != ''
    """).fetchone()[0]
    print(f"维度标签覆盖: {tagged}/{valid} ({tagged/valid*100:.1f}%)")

    # 数据库文件大小
    db_size = os.path.getsize(db_path)
    print(f"\n数据库文件: {db_size/1024:.1f} KB")

    conn.close()


def main():
    parser = argparse.ArgumentParser(description="SQLite 入库")
    parser.add_argument("--csv", type=str, help="清洗后CSV路径")
    parser.add_argument("--stats", action="store_true", help="仅打印统计")
    args = parser.parse_args()

    db_path = config.SQLITE_DB
    csv_path = args.csv or os.path.join(config.DATA_DIR, "cleaned_comments.csv")

    if args.stats:
        if os.path.exists(db_path):
            print_db_stats(db_path)
        else:
            print("❌ 数据库不存在，请先运行导入")
        return

    # 检查CSV
    if not os.path.exists(csv_path):
        print(f"❌ 找不到清洗后CSV: {csv_path}")
        print("   请先运行: python scripts/step1_clean.py")
        sys.exit(1)

    # 建表
    create_tables(db_path)

    # 导入
    load_cleaned_csv(csv_path, db_path)

    # 记录采集日志
    record_crawl_log(csv_path, db_path)

    # 统计
    print_db_stats(db_path)


if __name__ == "__main__":
    main()
