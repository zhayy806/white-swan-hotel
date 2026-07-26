#!/usr/bin/env python3
"""
OTA 酒店评论智能分析系统 — FastAPI 服务
========================================
启动: python scripts/step4_serve.py
访问: http://localhost:8000

接口一览:
  GET  /api/stats           — 统计总览
  GET  /api/comments        — 评论分页查询（多条件筛选）
  GET  /api/sentiment       — 情感分布
  GET  /api/dimensions      — 观点维度统计
  GET  /api/room-types      — 房型对比
  GET  /api/crawl-log       — 采集日志
  POST /api/rag/ask         — RAG 智能问答（双路检索）
  GET  /api/export          — 导出Excel
  GET  /                    — 前端可视化看板
"""

import os
import sys
import sqlite3
import json
import io
from datetime import datetime
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import glob
import pandas as pd

import config

# ============================================================
# FastAPI 初始化
# ============================================================
app = FastAPI(
    title="OTA酒店评论智能分析系统",
    description="白天鹅宾馆评论数据分析 + RAG智能问答",
    version="1.0.0",
)

# 静态文件（酒店实景图片）
os.makedirs(os.path.join(config.PROJECT_DIR, "static", "images"), exist_ok=True)
app.mount("/static", StaticFiles(directory=os.path.join(config.PROJECT_DIR, "static")), name="static")

# ============================================================
# 图片映射：FAQ主题 → 推荐展示的图片
# ============================================================
IMAGE_MAPPING = {
    # 主题关键词 → [图片文件名列表]（与 static/images/ 下中文命名对应）
    "酒店概况|大堂|故乡水|瀑布":         ["images/lobby-waterfall.jpg", "images/大堂全景.jpg"],
    "入住|退房|前台|押金":               ["images/大堂全景.jpg"],
    "江景|珠江|夜景|视野":               ["images/酒店外观·珠江畔.jpg", "images/珠江夜景.jpg", "images/豪华江景大床房.jpg"],
    "儿童|亲子|加床|小孩|婴儿":          ["images/儿童洗漱用品.jpg", "images/标准客房.jpg"],
    "一次性用品|洗漱|欧舒丹|拖鞋":       ["images/豪华江景大床房.jpg"],
    "客房|房间|空调|WiFi|迷你吧":        ["images/豪华江景大床房.jpg", "images/标准客房.jpg"],
    "玉堂春暖|米其林|粤菜|葵花鸡":       ["images/玉堂春暖·米其林.jpg"],
    "宏图府|早茶|点心|虾饺|烧卖":       ["images/玉堂春暖·米其林.jpg"],
    "流浮阁|自助餐|早餐|西餐":           ["images/流浮阁自助餐.jpg"],
    "行政酒廊|下午茶|鸡尾酒|Happy Hour": ["images/行政酒廊.jpg"],
    "泳池|无边|恒温|按摩池|干蒸":       ["images/户外江景泳池.jpg", "images/珠江夜景.jpg"],
    "健身|健身房|跑步机":                ["images/健身中心.jpg"],
    "SPA|水疗|按摩|推拿|精油":           ["images/水疗中心.jpg"],
    "停车|充电桩|车位":                  ["images/停车场.jpg"],
    "沙面|周边|欧式|建筑|拍照":          ["images/沙面欧式建筑.jpg"],
    "花园|锦鲤|园林|园艺|鹿角树":        ["images/花园锦鲤池塘.webp"],
    "文创|伴手礼|月饼|明信片":           ["images/文创产品.jpg"],
    "旅拍|摄影|精修|预约":               ["images/沙面欧式建筑.jpg"],
    "套房|行政尊贵":                      ["images/行政套房.jpg"],
}

def find_images_for_question(question: str, faq_sections: list[str]) -> list[str]:
    """根据问题内容和命中的FAQ章节，推荐展示图片（不检查文件是否存在，前端优雅降级）"""
    matched = []
    seen = set()
    search_text = question + " " + " ".join(faq_sections)

    for pattern, images in IMAGE_MAPPING.items():
        keywords = pattern.split("|")
        if any(kw in search_text for kw in keywords):
            for img in images:
                if img not in seen:
                    matched.append(img)
                    seen.add(img)

    # 没有匹配 → 默认酒店外观
    if not matched:
        matched.append("images/酒店外观·珠江畔.jpg")

    return matched[:4]  # 最多4张

# ============================================================
# 数据库工具
# ============================================================

def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(config.SQLITE_DB)
    conn.row_factory = sqlite3.Row
    return conn


def db_to_json(rows: list) -> list:
    return [dict(r) for r in rows]


# ============================================================
# Lazy-load RAG 组件（首次调用时加载，避免启动慢）
# ============================================================
_rag = None

def get_rag():
    global _rag
    if _rag is None:
        from scripts.onnx_embed import ONNXEmbeddings
        from langchain_community.vectorstores import Chroma
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.runnables import RunnablePassthrough
        from langchain_core.output_parsers import StrOutputParser

        embedding = ONNXEmbeddings(config.ONNX_MODEL_DIR)
        vs = Chroma(persist_directory=config.CHROMA_DIR, embedding_function=embedding)

        llm = None
        if config.LLM_API_KEY:
            llm = ChatOpenAI(
                base_url=config.LLM_BASE_URL,
                api_key=config.LLM_API_KEY,
                model=config.LLM_MODEL,
                temperature=0.3,
                max_tokens=1024,
            )

        PROMPT = """你是广州白天鹅宾馆的AI助手。请依据以下信息回答用户问题。

🏨 酒店官方资料：
---
{faq_context}
---

💬 住客真实评论：
---
{review_context}
---

用户问题：{question}

要求：
1. 政策类问题以官方资料为准，用评论佐证
2. 体验类问题以评论为准，官方资料补充背景
3. 如果都没覆盖，诚实说明
4. 引用来源标注「官方资料」或「住客评论(日期)」
5. 客观呈现好评和差评两个方面

回答："""

        chain = None
        if llm:
            prompt = ChatPromptTemplate.from_template(PROMPT)
            chain = prompt | llm | StrOutputParser()

        _rag = {
            "vectorstore": vs,
            "llm": llm,
            "chain": chain,
        }
    return _rag


# ============================================================
# Pydantic 模型
# ============================================================

class RAGRequest(BaseModel):
    question: str
    faq_k: int = 3
    review_k: int = 5
    use_llm: bool = True


class RAGResponse(BaseModel):
    question: str
    answer: str
    faq_sources: list
    review_sources: list
    images: list[str] = []  # 推荐展示的图片路径


# ============================================================
# API: 统计总览
# ============================================================

@app.get("/api/stats")
def get_stats():
    conn = get_db()
    total = conn.execute("SELECT COUNT(*) FROM comment").fetchone()[0]
    valid = conn.execute("SELECT COUNT(*) FROM comment WHERE is_valid=1").fetchone()[0]
    avg_rating = conn.execute(
        "SELECT ROUND(AVG(rating_score),2) FROM comment WHERE is_valid=1"
    ).fetchone()[0]
    date_range = conn.execute(
        "SELECT MIN(review_date), MAX(review_date) FROM comment WHERE is_valid=1"
    ).fetchone()

    sentiment = db_to_json(conn.execute("""
        SELECT sentiment, COUNT(*) as cnt, ROUND(AVG(rating_score),2) as avg_score
        FROM comment WHERE is_valid=1
        GROUP BY sentiment ORDER BY avg_score DESC
    """).fetchall())

    top_room = db_to_json(conn.execute("""
        SELECT room_type, COUNT(*) as cnt, ROUND(AVG(rating_score),2) as avg_score
        FROM comment WHERE is_valid=1
        GROUP BY room_type ORDER BY cnt DESC LIMIT 5
    """).fetchall())

    travel = db_to_json(conn.execute("""
        SELECT travel_type, COUNT(*) as cnt FROM comment WHERE is_valid=1
        GROUP BY travel_type ORDER BY cnt DESC
    """).fetchall())

    conn.close()
    return {
        "total": total,
        "valid": valid,
        "avg_rating": avg_rating,
        "date_range": [date_range[0], date_range[1]],
        "sentiment": sentiment,
        "top_room_types": top_room,
        "travel_types": travel,
    }


# ============================================================
# API: 评论分页查询
# ============================================================

@app.get("/api/comments")
def get_comments(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    sentiment: Optional[str] = None,
    room_type: Optional[str] = None,
    travel_type: Optional[str] = None,
    rating_min: Optional[float] = None,
    rating_max: Optional[float] = None,
    keyword: Optional[str] = None,
    dimension: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    valid_only: bool = True,
):
    conn = get_db()
    conditions = []
    params = []

    if valid_only:
        conditions.append("is_valid = 1")
    if sentiment:
        conditions.append("sentiment = ?")
        params.append(sentiment)
    if room_type:
        conditions.append("room_type = ?")
        params.append(room_type)
    if travel_type:
        conditions.append("travel_type = ?")
        params.append(travel_type)
    if rating_min is not None:
        conditions.append("rating_score >= ?")
        params.append(rating_min)
    if rating_max is not None:
        conditions.append("rating_score <= ?")
        params.append(rating_max)
    if keyword:
        conditions.append("comment_text LIKE ?")
        params.append(f"%{keyword}%")
    if dimension:
        conditions.append("dimension_tags LIKE ?")
        params.append(f"%{dimension}%")
    if date_from:
        conditions.append("review_date >= ?")
        params.append(date_from)
    if date_to:
        conditions.append("review_date <= ?")
        params.append(date_to)

    where = "WHERE " + " AND ".join(conditions) if conditions else ""

    # 总数
    total = conn.execute(f"SELECT COUNT(*) FROM comment {where}", params).fetchone()[0]

    # 分页
    offset = (page - 1) * page_size
    rows = conn.execute(
        f"""SELECT comment_id, user_id, comment_text, room_type, travel_type,
                   rating_score, rating_label, review_date, sentiment, dimension_tags
            FROM comment {where}
            ORDER BY comment_id DESC
            LIMIT ? OFFSET ?""",
        params + [page_size, offset],
    ).fetchall()

    # 筛选下拉选项
    room_options = [r[0] for r in conn.execute(
        "SELECT DISTINCT room_type FROM comment WHERE is_valid=1 ORDER BY room_type"
    ).fetchall()]
    travel_options = [r[0] for r in conn.execute(
        "SELECT DISTINCT travel_type FROM comment WHERE is_valid=1 ORDER BY travel_type"
    ).fetchall()]

    conn.close()
    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": (total + page_size - 1) // page_size,
        "data": db_to_json(rows),
        "filters": {
            "room_types": room_options,
            "travel_types": travel_options,
            "sentiments": ["好评", "中评", "差评"],
            "dimensions": list(config.DIMENSION_KEYWORDS.keys()),
        },
    }


# ============================================================
# API: 情感分布
# ============================================================

@app.get("/api/sentiment")
def get_sentiment():
    conn = get_db()
    data = db_to_json(conn.execute("""
        SELECT sentiment, COUNT(*) as cnt, ROUND(AVG(rating_score),2) as avg
        FROM comment WHERE is_valid=1
        GROUP BY sentiment ORDER BY avg DESC
    """).fetchall())

    # 月趋势
    monthly = db_to_json(conn.execute("""
        SELECT review_date, sentiment, COUNT(*) as cnt
        FROM comment WHERE is_valid=1 AND review_date >= '2024-01'
        GROUP BY review_date, sentiment
        ORDER BY review_date
    """).fetchall())

    conn.close()
    return {"distribution": data, "monthly_trend": monthly}


# ============================================================
# API: 观点维度统计
# ============================================================

@app.get("/api/dimensions")
def get_dimensions():
    conn = get_db()
    dim_stats = {}
    for dim in config.DIMENSION_KEYWORDS:
        total = conn.execute(
            "SELECT COUNT(*) FROM comment WHERE is_valid=1 AND dimension_tags LIKE ?",
            (f"%{dim}%",),
        ).fetchone()[0]
        avg_score = conn.execute(
            "SELECT ROUND(AVG(rating_score),2) FROM comment WHERE is_valid=1 AND dimension_tags LIKE ?",
            (f"%{dim}%",),
        ).fetchone()[0]
        neg_count = conn.execute(
            "SELECT COUNT(*) FROM comment WHERE is_valid=1 AND dimension_tags LIKE ? AND sentiment='差评'",
            (f"%{dim}%",),
        ).fetchone()[0]
        dim_stats[dim] = {
            "total": total,
            "avg_score": avg_score or 0,
            "negative": neg_count,
            "negative_rate": round(neg_count / total * 100, 1) if total > 0 else 0,
        }

    conn.close()
    return dim_stats


# ============================================================
# API: 房型对比
# ============================================================

@app.get("/api/room-types")
def get_room_types():
    conn = get_db()
    data = db_to_json(conn.execute("""
        SELECT room_type, COUNT(*) as cnt, ROUND(AVG(rating_score),2) as avg_score,
               SUM(CASE WHEN sentiment='好评' THEN 1 ELSE 0 END) as positive,
               SUM(CASE WHEN sentiment='中评' THEN 1 ELSE 0 END) as neutral,
               SUM(CASE WHEN sentiment='差评' THEN 1 ELSE 0 END) as negative
        FROM comment WHERE is_valid=1
        GROUP BY room_type ORDER BY cnt DESC
    """).fetchall())
    conn.close()
    return data


# ============================================================
# API: 采集日志
# ============================================================

@app.get("/api/crawl-log")
def get_crawl_log():
    conn = get_db()
    rows = db_to_json(conn.execute(
        "SELECT * FROM crawl_log ORDER BY created_at DESC LIMIT 20"
    ).fetchall())
    conn.close()
    return rows


# ============================================================
# API: RAG 智能问答（双路检索）
# ============================================================

@app.post("/api/rag/ask", response_model=RAGResponse)
def rag_ask(req: RAGRequest):
    rag = get_rag()
    vs = rag["vectorstore"]

    # 双路检索：FAQ + 评论
    faq_docs = vs.similarity_search(req.question, k=req.faq_k, filter={"source": "hotel_faq"})
    review_docs = vs.similarity_search(req.question, k=req.review_k, filter={"source": "user_review"})

    # 格式化FAQ来源
    faq_sources = []
    for d in faq_docs:
        faq_sources.append({
            "section": d.metadata.get("section", ""),
            "keywords": d.metadata.get("topic_keywords", ""),
            "content": d.page_content[:300],
        })

    # 格式化评论来源
    review_sources = []
    for d in review_docs:
        m = d.metadata
        review_sources.append({
            "comment_id": m.get("comment_id"),
            "review_date": m.get("review_date", ""),
            "room_type": m.get("room_type", ""),
            "rating_score": m.get("rating_score"),
            "sentiment": m.get("sentiment", ""),
            "dimension_tags": m.get("dimension_tags", ""),
            "content": d.page_content[:300],
        })

    # LLM 生成
    answer = ""
    if req.use_llm and rag["chain"]:
        faq_ctx = "\n\n".join(
            f"[官方资料-{d.metadata.get('section','')}]\n{d.page_content}"
            for d in faq_docs
        )
        rev_ctx = "\n\n".join(
            f"[住客评论] {d.metadata.get('review_date','')} | "
            f"{d.metadata.get('room_type','')} | 评分{d.metadata.get('rating_score','')} | "
            f"{d.metadata.get('sentiment','')}\n{d.page_content}"
            for d in review_docs
        )
        try:
            answer = rag["chain"].invoke({
                "faq_context": faq_ctx or "无相关官方信息",
                "review_context": rev_ctx or "无相关住客评论",
                "question": req.question,
            })
        except Exception as e:
            answer = f"AI回答生成失败: {str(e)}"
    elif not rag["llm"]:
        answer = "（未配置LLM API Key，仅展示检索结果。设置环境变量 DEEPSEEK_API_KEY 启用AI回答）"

    # 匹配相关图片
    faq_section_names = [s["section"] for s in faq_sources]
    images = find_images_for_question(req.question, faq_section_names)

    return RAGResponse(
        question=req.question,
        answer=answer,
        faq_sources=faq_sources,
        review_sources=review_sources,
        images=images,
    )


# ============================================================
# API: 导出Excel报告
# ============================================================

@app.get("/api/export")
def export_excel(
    sentiment: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
):
    conn = get_db()
    conditions = ["is_valid = 1"]
    params = []
    if sentiment:
        conditions.append("sentiment = ?")
        params.append(sentiment)
    if date_from:
        conditions.append("review_date >= ?")
        params.append(date_from)
    if date_to:
        conditions.append("review_date <= ?")
        params.append(date_to)

    where = "WHERE " + " AND ".join(conditions)
    df = pd.read_sql_query(
        f"""SELECT comment_id, user_id, comment_text, room_type, travel_type,
                   rating_score, rating_label, review_date, sentiment, dimension_tags
            FROM comment {where} ORDER BY comment_id DESC""",
        conn,
    )
    conn.close()

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="评论数据", index=False)
        # 统计sheet
        summary = pd.DataFrame([
            {"指标": "总评论数", "值": len(df)},
            {"指标": "平均评分", "值": round(df["rating_score"].mean(), 2)},
            {"指标": "好评数", "值": (df["sentiment"] == "好评").sum()},
            {"指标": "中评数", "值": (df["sentiment"] == "中评").sum()},
            {"指标": "差评数", "值": (df["sentiment"] == "差评").sum()},
        ])
        summary.to_excel(writer, sheet_name="统计摘要", index=False)
    output.seek(0)

    filename = f"白天鹅宾馆评论报告_{datetime.now().strftime('%Y%m%d')}.xlsx"
    return StreamingResponse(
        output,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


# ============================================================
# 前端看板（单HTML文件）
# ============================================================

FRONTEND_HTML = r"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>白天鹅宾馆 · 智能管家</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
/* ============================================================
   白天鹅宾馆品牌设计系统
   配色: 珠江水蓝 #0a2849 / 鎏金 #c9a96e / 天鹅白 #faf8f5
   ============================================================ */

@import url('https://fonts.googleapis.com/css2?family=Noto+Serif+SC:wght@400;600;700&family=Noto+Sans+SC:wght@300;400;500&display=swap');

*{margin:0;padding:0;box-sizing:border-box}

:root {
  --deep-navy: #0a1f35;
  --water-blue: #1a3f5c;
  --gold: #c9a96e;
  --gold-light: #e0d0a8;
  --cream: #faf8f5;
  --white: #ffffff;
  --text: #2d2d2d;
  --text-light: #6b7280;
  --text-muted: #9ca3af;
  --border: #e8e4dd;
  --shadow: 0 2px 20px rgba(10,31,53,0.08);
  --shadow-lg: 0 8px 40px rgba(10,31,53,0.12);
  --radius: 16px;
  --radius-sm: 10px;
  --transition: 0.3s cubic-bezier(0.4,0,0.2,1);
}

body {
  font-family: 'Noto Sans SC', -apple-system, sans-serif;
  background: var(--cream);
  color: var(--text);
  line-height: 1.6;
  -webkit-font-smoothing: antialiased;
}

/* ---- Hero 区域 ---- */
.hero {
  position: relative;
  min-height: 580px;
  display: flex;
  align-items: flex-end;
  overflow: hidden;
  background: #1a3f5c url('/static/images/lobby-waterfall.jpg') center center / cover no-repeat;
  /* 图片未加载时回退到深蓝渐变 */
  background-color: #1a3f5c;
}

/* 图片上的半透明叠加层（保证文字可读，但图片清晰可见） */
.hero::before {
  content: '';
  position: absolute;
  inset: 0;
  background: linear-gradient(
    to bottom,
    rgba(10,25,40,0.0) 0%,
    rgba(10,25,40,0.05) 40%,
    rgba(10,25,40,0.2) 75%,
    rgba(10,25,40,0.4) 100%
  );
  z-index: 0;
}

/* 底部波浪过渡到奶油色内容区 */
.hero::after {
  content: '';
  position: absolute;
  bottom: -2px;
  left: 0;
  right: 0;
  height: 80px;
  background: var(--cream);
  clip-path: ellipse(75% 100% at 50% 100%);
  z-index: 1;
}

.hero-content {
  position: relative;
  z-index: 2;
  text-align: center;
  padding: 0 24px 60px;
  max-width: 760px;
  margin: 0 auto;
  width: 100%;
}

.hero-badge {
  display: inline-block;
  padding: 6px 18px;
  border: 1px solid rgba(201,169,110,0.5);
  border-radius: 50px;
  color: var(--gold-light);
  font-size: 12px;
  letter-spacing: 4px;
  text-transform: uppercase;
  margin-bottom: 24px;
  backdrop-filter: blur(10px);
  background: rgba(10,25,40,0.3);
}

.hero h1 {
  font-family: 'Noto Serif SC', serif;
  font-size: clamp(36px, 6vw, 56px);
  font-weight: 700;
  color: var(--white);
  margin-bottom: 12px;
  letter-spacing: 4px;
  text-shadow: 0 2px 20px rgba(0,0,0,0.3);
}

.hero h1 em {
  font-style: normal;
  color: var(--gold);
}

.hero .hero-tagline {
  color: rgba(255,255,255,0.8);
  font-size: 17px;
  font-weight: 300;
  margin-bottom: 28px;
  letter-spacing: 2px;
  text-shadow: 0 1px 8px rgba(0,0,0,0.3);
}

.hero-intro {
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  gap: 20px;
  font-size: 13px;
  color: rgba(255,255,255,0.65);
  letter-spacing: 1px;
}

.hero-intro span {
  display: flex;
  align-items: center;
  gap: 6px;
}

.hero-intro .divider {
  color: rgba(201,169,110,0.4);
}

/* ---- 主体布局 ---- */
.container {
  max-width: 880px;
  margin: 0 auto;
  padding: 0 24px 60px;
  position: relative;
  z-index: 1;
}

/* ---- 问答核心区 ---- */
.qa-section {
  margin-top: -40px;
  position: relative;
  z-index: 2;
}

.qa-card {
  background: var(--white);
  border-radius: var(--radius);
  box-shadow: var(--shadow-lg);
  padding: 36px;
  margin-bottom: 24px;
}

.qa-card h2 {
  font-family: 'Noto Serif SC', serif;
  font-size: 22px;
  color: var(--deep-navy);
  margin-bottom: 8px;
}

.qa-card .qa-subtitle {
  font-size: 14px;
  color: var(--text-light);
  margin-bottom: 28px;
}

.qa-input-row {
  display: flex;
  gap: 12px;
  margin-bottom: 20px;
}

.qa-input-row input {
  flex: 1;
  padding: 16px 20px;
  border: 2px solid var(--border);
  border-radius: var(--radius-sm);
  font-size: 15px;
  font-family: inherit;
  outline: none;
  transition: var(--transition);
  background: var(--cream);
}

.qa-input-row input:focus {
  border-color: var(--gold);
  background: var(--white);
  box-shadow: 0 0 0 4px rgba(201,169,110,0.1);
}

.qa-input-row button {
  padding: 16px 32px;
  background: linear-gradient(135deg, #c9a96e, #b8943f);
  color: var(--white);
  border: none;
  border-radius: var(--radius-sm);
  font-size: 15px;
  font-weight: 500;
  cursor: pointer;
  transition: var(--transition);
  white-space: nowrap;
  font-family: inherit;
  letter-spacing: 1px;
}

.qa-input-row button:hover {
  transform: translateY(-1px);
  box-shadow: 0 4px 16px rgba(201,169,110,0.35);
}


/* ---- 回答卡片 ---- */
.answer-wrapper {
  display: none;
  margin-top: 24px;
  animation: fadeIn 0.4s ease;
}

@keyframes fadeIn {
  from { opacity: 0; transform: translateY(8px); }
  to { opacity: 1; transform: translateY(0); }
}

.answer-header {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 16px;
}

.answer-header .dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: var(--gold);
  animation: pulse 2s ease-in-out infinite;
}

@keyframes pulse {
  0%, 100% { opacity: 1; transform: scale(1); }
  50% { opacity: 0.5; transform: scale(0.8); }
}

.answer-header span {
  font-size: 13px;
  color: var(--text-muted);
  letter-spacing: 1px;
}

.answer-content {
  background: linear-gradient(135deg, #fdfbf7 0%, #f8f4ec 100%);
  border: 1px solid #efe8d8;
  border-radius: var(--radius-sm);
  padding: 24px 28px;
  font-size: 15px;
  line-height: 1.9;
  color: var(--text);
  white-space: pre-wrap;
  position: relative;
}

.answer-content::before {
  content: '"';
  position: absolute;
  top: -8px;
  left: 20px;
  font-family: 'Noto Serif SC', serif;
  font-size: 48px;
  color: var(--gold);
  opacity: 0.2;
  line-height: 1;
}

/* ---- 来源卡片 ---- */
.sources-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 16px;
  margin-top: 16px;
}

.source-col h4 {
  font-size: 12px;
  color: var(--text-muted);
  letter-spacing: 2px;
  text-transform: uppercase;
  margin-bottom: 10px;
  font-weight: 500;
}

.source-chip {
  background: var(--white);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  padding: 12px 14px;
  margin-bottom: 8px;
  font-size: 13px;
  line-height: 1.5;
  transition: var(--transition);
}

.source-chip:hover {
  border-color: var(--gold-light);
  box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}

.source-chip .src-meta {
  font-size: 11px;
  color: var(--text-muted);
  margin-bottom: 4px;
  display: flex;
  align-items: center;
  gap: 6px;
}

.src-tag {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 20px;
  font-size: 10px;
  letter-spacing: 1px;
}

.src-tag.faq { background: #fef3c7; color: #92400e; }
.src-tag.positive { background: #dcfce7; color: #166534; }
.src-tag.negative { background: #fef2f2; color: #991b1b; }
.src-tag.neutral { background: #f0fdf4; color: #15803d; }

.answer-loading {
  display: flex;
  align-items: center;
  gap: 12px;
  color: var(--text-muted);
  font-size: 14px;
  padding: 24px;
}

/* ---- 实景图片画廊 ---- */
.answer-images {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 10px;
  margin-bottom: 20px;
}

.answer-images.single { grid-template-columns: 1fr; max-width: 500px; }
.answer-images.double { grid-template-columns: 1fr 1fr; }
.answer-images.triple { grid-template-columns: repeat(3, 1fr); }

.img-card {
  position: relative;
  border-radius: var(--radius-sm);
  overflow: hidden;
  cursor: pointer;
  aspect-ratio: 16/10;
  background: #e8e4dd;
  transition: var(--transition);
}

.img-card:hover {
  transform: scale(1.02);
  box-shadow: 0 4px 20px rgba(0,0,0,0.15);
}

.img-card img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
}

.img-card .img-label {
  position: absolute;
  bottom: 0;
  left: 0;
  right: 0;
  padding: 8px 12px;
  background: linear-gradient(transparent, rgba(0,0,0,0.6));
  color: #fff;
  font-size: 11px;
  letter-spacing: 1px;
}

/* 图片占位（未放照片前） */
.img-placeholder {
  width: 100%;
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-direction: column;
  gap: 8px;
  background: linear-gradient(135deg, #e8e4dd, #d5cfc4);
  color: #9ca3af;
  font-size: 12px;
}

.img-placeholder .ph-icon {
  font-size: 32px;
  opacity: 0.5;
}

/* 灯箱 */
.lightbox {
  display: none;
  position: fixed;
  inset: 0;
  background: rgba(10,31,53,0.95);
  z-index: 9999;
  align-items: center;
  justify-content: center;
  cursor: pointer;
}

.lightbox.active { display: flex; }

.lightbox img {
  max-width: 90vw;
  max-height: 85vh;
  border-radius: 8px;
  box-shadow: 0 8px 60px rgba(0,0,0,0.5);
}

.lightbox .close {
  position: absolute;
  top: 20px;
  right: 30px;
  color: #fff;
  font-size: 32px;
  cursor: pointer;
  opacity: 0.6;
  transition: opacity 0.2s;
}

.lightbox .close:hover { opacity: 1; }

.answer-loading .spinner {
  width: 20px;
  height: 20px;
  border: 2px solid var(--border);
  border-top-color: var(--gold);
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

/* ---- 数据面板 ---- */
.data-section {
  margin-top: 16px;
}

.data-section h2 {
  font-family: 'Noto Serif SC', serif;
  font-size: 20px;
  color: var(--deep-navy);
  margin-bottom: 6px;
}

.data-section .data-subtitle {
  font-size: 14px;
  color: var(--text-light);
  margin-bottom: 24px;
}

.stat-row {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 16px;
  margin-bottom: 24px;
}

.stat-card {
  background: var(--white);
  border-radius: var(--radius-sm);
  padding: 20px;
  box-shadow: var(--shadow);
  transition: var(--transition);
}

.stat-card:hover {
  transform: translateY(-2px);
  box-shadow: var(--shadow-lg);
}

.stat-card .stat-icon {
  font-size: 24px;
  margin-bottom: 8px;
}

.stat-card .stat-num {
  font-family: 'Noto Serif SC', serif;
  font-size: 26px;
  font-weight: 700;
  color: var(--deep-navy);
}

.stat-card .stat-label {
  font-size: 12px;
  color: var(--text-muted);
  margin-top: 2px;
}

.chart-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 16px;
  margin-bottom: 24px;
}

.chart-card {
  background: var(--white);
  border-radius: var(--radius-sm);
  padding: 24px;
  box-shadow: var(--shadow);
}

.chart-card h3 {
  font-size: 14px;
  color: var(--text-light);
  margin-bottom: 16px;
  font-weight: 500;
  letter-spacing: 1px;
}

.chart-wrap {
  position: relative;
  height: 220px;
}

.chart-wrap canvas { width: 100% !important; }

/* 评论预览流 */
.review-stream {
  background: var(--white);
  border-radius: var(--radius-sm);
  box-shadow: var(--shadow);
  padding: 24px;
  margin-bottom: 24px;
}

.review-stream h3 {
  font-size: 14px;
  color: var(--text-light);
  margin-bottom: 16px;
  font-weight: 500;
  letter-spacing: 1px;
}

.review-cards {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
  gap: 12px;
}

.review-mini {
  background: var(--cream);
  border-radius: var(--radius-sm);
  padding: 14px;
  font-size: 13px;
  line-height: 1.6;
  transition: var(--transition);
  cursor: default;
}

.review-mini:hover {
  background: #f0ebe0;
}

.review-mini .rm-meta {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 8px;
  font-size: 11px;
  color: var(--text-muted);
}

.review-mini .rm-rating {
  font-weight: 600;
  color: var(--gold);
}

/* ---- Footer ---- */
.footer {
  text-align: center;
  padding: 20px;
  color: var(--text-muted);
  font-size: 12px;
  letter-spacing: 1px;
}

/* ---- Responsive ---- */
@media (max-width: 768px) {
  .stat-row { grid-template-columns: repeat(2, 1fr); }
  .chart-grid { grid-template-columns: 1fr; }
  .sources-grid { grid-template-columns: 1fr; }
  .qa-input-row { flex-direction: column; }
  .qa-card { padding: 24px; }
  .hero-stats { gap: 20px; }
}
</style>
</head>
<body>

<!-- ============================================================ -->
<!-- HERO -->
<!-- ============================================================ -->
<section class="hero">
  <div class="hero-content">
    <div class="hero-badge">Since 1983 · Five Stars</div>
    <h1>广州<em>白天鹅</em>宾馆</h1>
    <p class="hero-tagline">中国第一家五星级酒店 · 岭南文化的精神地标</p>
    <div class="hero-intro">
      <span>🏛️ 沙面岛 · 珠江水畔</span>
      <span class="divider">|</span>
      <span>📐 莫伯治 · 佘畯南大师设计</span>
      <span class="divider">|</span>
      <span>🌊 故乡水 · 濯月亭</span>
      <span class="divider">|</span>
      <span>🍽️ 米其林一星 · 玉堂春暖</span>
      <span class="divider">|</span>
      <span>🏊 恒温江景无边际泳池</span>
      <span class="divider">|</span>
      <span>🏆 中国20世纪建筑遗产</span>
    </div>
  </div>
</section>

<!-- ============================================================ -->
<!-- 智能管家 Q&A -->
<!-- ============================================================ -->
<div class="container">
<div class="qa-section">
  <div class="qa-card">
    <h2>🦢 您的专属智能管家</h2>
    <p class="qa-subtitle">我熟知酒店的每一个细节。关于白天鹅，您想了解什么？</p>

    <div class="qa-input-row">
      <input id="rag-question" type="text"
             placeholder="输入您想了解的问题，管家为您解答..."
             onkeydown="if(event.key==='Enter')askRAG()">
      <button onclick="askRAG()">询 问 管 家</button>
    </div>


    <!-- 回答区 -->
    <div class="answer-wrapper" id="answer-wrapper">
      <div class="answer-header"><div class="dot"></div><span>您的专属智能管家</span></div>

      <!-- 实景图片展示 -->
      <div class="answer-images" id="answer-images" style="display:none"></div>

      <div class="answer-content" id="rag-answer"></div>
      <div class="sources-grid" id="rag-sources" style="display:none">
        <div class="source-col"><h4>📋 酒店官方信息</h4><div id="faq-sources"></div></div>
        <div class="source-col"><h4>💬 住客真实评价</h4><div id="review-sources"></div></div>
      </div>
    </div>
  </div>
</div>

<!-- ============================================================ -->
<!-- 数据一览 -->
<!-- ============================================================ -->
<div class="data-section">
  <h2>📊 住客真实反馈</h2>
  <p class="data-subtitle">住客真实反馈一览</p>

  <div class="stat-row">
    <div class="stat-card">
      <div class="stat-icon">⭐</div>
      <div class="stat-num" id="stat-rating">4.80</div>
      <div class="stat-label">综合平均评分</div>
    </div>
    <div class="stat-card">
      <div class="stat-icon">😊</div>
      <div class="stat-num" id="stat-pos-rate" style="color:#166534">95.4%</div>
      <div class="stat-label">好评率（4.0+）</div>
    </div>
    <div class="stat-card">
      <div class="stat-icon">😐</div>
      <div class="stat-num" id="stat-neu" style="color:#92400e">—</div>
      <div class="stat-label">中评数</div>
    </div>
    <div class="stat-card">
      <div class="stat-icon">😞</div>
      <div class="stat-num" id="stat-neg" style="color:#991b1b">—</div>
      <div class="stat-label">差评数</div>
    </div>
  </div>

  <div class="chart-grid">
    <div class="chart-card">
      <h3>情感分布</h3>
      <div class="chart-wrap"><canvas id="sentimentChart"></canvas></div>
    </div>
    <div class="chart-card">
      <h3>月度评分趋势</h3>
      <div class="chart-wrap"><canvas id="trendChart"></canvas></div>
    </div>
  </div>

  <!-- 住客真实声音 -->
  <div class="review-stream">
    <h3>💬 住客真实声音</h3>
    <div class="review-cards" id="review-cards"></div>
  </div>

</div>

<div class="footer">
  广州白天鹅宾馆 · 沙面南街 1 号 · 智能管家由 AI 驱动 · 数据来源真实住客评论
</div>
</div>

<!-- ============================================================ -->
<!-- Scripts -->
<!-- ============================================================ -->
<script>
// ---- 工具 ----
async function api(url, opts) {
  const r = await fetch(url, opts);
  if (!r.ok) throw new Error(r.statusText);
  return r.json();
}

// ---- 加载统计 ----
(async function init() {
  const d = await api('/api/stats');
  const pos = d.sentiment.find(s=>s.sentiment==='好评')?.cnt||0;
  // Stat cards
  document.getElementById('stat-rating').textContent = d.avg_rating;
  document.getElementById('stat-pos-rate').textContent = (pos/d.valid*100).toFixed(1)+'%';
  const neu = d.sentiment.find(s=>s.sentiment==='中评')?.cnt||0;
  const neg = d.sentiment.find(s=>s.sentiment==='差评')?.cnt||0;
  document.getElementById('stat-neu').textContent = neu;
  document.getElementById('stat-neg').textContent = neg;

  // Sentiment doughnut
  const sc = d.sentiment;
  new Chart(document.getElementById('sentimentChart'), {
    type: 'doughnut',
    data: {
      labels: sc.map(x=>x.sentiment),
      datasets: [{data: sc.map(x=>x.cnt),
        backgroundColor: ['#c9a96e','#d4c5a0','#e0d4bc'],
        borderColor: '#fff', borderWidth: 2}]
    },
    options: {
      responsive:true, maintainAspectRatio:false,
      cutout:'70%',
      plugins: { legend: { position:'bottom', labels:{padding:20,usePointStyle:true,pointStyleWidth:8} }}
    }
  });

  // Trend
  const trend = await api('/api/sentiment');
  const months = [...new Set(trend.monthly_trend.map(x=>x.review_date))].sort();
  const bySent = {};
  trend.monthly_trend.forEach(x => {
    if(!bySent[x.sentiment]) bySent[x.sentiment] = {};
    bySent[x.sentiment][x.review_date] = x.cnt;
  });
  new Chart(document.getElementById('trendChart'), {
    type: 'line',
    data: {
      labels: months,
      datasets: [
        {label:'好评',data:months.map(m=>bySent['好评']?.[m]||0),
         borderColor:'#c9a96e',backgroundColor:'rgba(201,169,110,0.08)',fill:true,tension:0.4,pointRadius:0},
        {label:'中评',data:months.map(m=>bySent['中评']?.[m]||0),
         borderColor:'#d4c5a0',fill:false,tension:0.4,pointRadius:0},
        {label:'差评',data:months.map(m=>bySent['差评']?.[m]||0),
         borderColor:'#e0d4bc',fill:false,tension:0.4,pointRadius:0},
      ]
    },
    options: {
      responsive:true, maintainAspectRatio:false,
      plugins:{legend:{position:'bottom',labels:{padding:20,usePointStyle:true,pointStyleWidth:8}}},
      scales:{y:{grid:{color:'#f0ebe0'},ticks:{font:{size:11}}},x:{grid:{display:false},ticks:{font:{size:10}}}}
    }
  });

  // Review stream
  const cmts = await api('/api/comments?page=1&page_size=8&valid_only=true');
  document.getElementById('review-cards').innerHTML = cmts.data.map(r => `
    <div class="review-mini">
      <div class="rm-meta">
        <span class="rm-rating">${'★'.repeat(Math.round(r.rating_score||0))}${'☆'.repeat(5-Math.round(r.rating_score||0))}</span>
        <span>${r.review_date||''}</span>
        <span>${(r.room_type||'').replace('豪华','').replace('行政','').substring(0,6)}</span>
      </div>
      ${(r.comment_text||'').substring(0,120)}${(r.comment_text||'').length>120?'...':''}
    </div>`).join('');
})();

// ---- 图片标签映射 ----
const IMG_LABELS = {
  'lobby-waterfall': '🏛️ 大堂 · 故乡水瀑布',
  '大堂全景': '🏛️ 大堂全景',
  '豪华江景大床房': '🛏️ 豪华江景大床房',
  '标准客房': '🛏️ 标准客房',
  '行政套房': '🛏️ 行政套房',
  '户外江景泳池': '🏊 恒温无边际江景泳池',
  '玉堂春暖·米其林': '🍽️ 玉堂春暖 · 米其林一星',
  '流浮阁自助餐': '🍳 流浮阁自助餐厅',
  '行政酒廊': '🥂 行政酒廊',
  '酒店外观·珠江畔': '🏨 酒店外观 · 珠江水畔',
  '沙面欧式建筑': '🏛️ 沙面岛欧式建筑群',
  '花园锦鲤池塘': '🌿 花园锦鲤池',
  '水疗中心': '💆 水疗中心',
  '健身中心': '🏋️ 健身中心',
  '儿童洗漱用品': '🧸 儿童专属用品',
  '文创产品': '🎁 白天鹅文创商店',
  '停车场': '🚗 停车场/充电桩',
  '珠江夜景': '🌃 珠江夜景',
};

function imageLabel(path) {
  const name = (path||'').split('/').pop().replace(/\.(jpg|png|webp|jpeg)$/i, '');
  return IMG_LABELS[name] || name;
}

function renderImages(images) {
  const div = document.getElementById('answer-images');
  if (!images || !images.length) { div.style.display = 'none'; return; }

  // 判断布局
  div.className = 'answer-images';
  if (images.length === 1) div.classList.add('single');
  else if (images.length === 2) div.classList.add('double');
  else if (images.length === 3) div.classList.add('triple');

  div.innerHTML = images.map(img => `
    <div class="img-card" onclick="openLightbox('/static/${img}')">
      <img src="/static/${img}" alt="${imageLabel(img)}"
           onerror="this.parentElement.innerHTML='<div class=\\'img-placeholder\\'><span class=\\'ph-icon\\'>🖼️</span>${imageLabel(img)}</div><div class=\\'img-label\\'>📷 待放入照片</div>'">
      <div class="img-label">${imageLabel(img)}</div>
    </div>
  `).join('');
  div.style.display = 'grid';
}

// ---- 灯箱 ----
function openLightbox(src) {
  let lb = document.getElementById('lightbox');
  if (!lb) {
    lb = document.createElement('div');
    lb.id = 'lightbox';
    lb.className = 'lightbox';
    lb.innerHTML = '<span class="close" onclick="closeLightbox()">&times;</span><img src="" alt="">';
    lb.addEventListener('click', function(e) { if (e.target === this) closeLightbox(); });
    document.body.appendChild(lb);
  }
  lb.querySelector('img').src = src;
  lb.classList.add('active');
  document.body.style.overflow = 'hidden';
}

function closeLightbox() {
  const lb = document.getElementById('lightbox');
  if (lb) { lb.classList.remove('active'); document.body.style.overflow = ''; }
}

// 键盘 ESC 关闭
document.addEventListener('keydown', function(e) {
  if (e.key === 'Escape') closeLightbox();
});

// ---- RAG 问答 ----
async function askRAG() {
  const q = document.getElementById('rag-question').value.trim();
  if (!q) return;
  document.getElementById('rag-question').value = q;

  const wrapper = document.getElementById('answer-wrapper');
  const answerDiv = document.getElementById('rag-answer');
  const sourcesDiv = document.getElementById('rag-sources');
  const imageDiv = document.getElementById('answer-images');

  wrapper.style.display = 'block';
  imageDiv.style.display = 'none';
  answerDiv.innerHTML = '<div class="answer-loading"><div class="spinner"></div>正在查阅酒店资料和住客评论...</div>';
  sourcesDiv.style.display = 'none';

  try {
    const r = await fetch('/api/rag/ask', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({question: q, faq_k: 3, review_k: 5, use_llm: true})
    });
    const d = await r.json();

    // 图片展示
    renderImages(d.images || []);

    // 美化回答
    let answer = d.answer || '抱歉，暂时无法回答这个问题。';
    answer = answer.replace(/\*\*(.+?)\*\*/g, '<strong style="color:#0a1f35">$1</strong>');
    answer = answer.replace(/^###?\s(.+)$/gm, '<span style="display:block;margin-top:16px;margin-bottom:6px;font-weight:600;color:#c9a96e;font-size:15px">$1</span>');
    answer = answer.replace(/^[-•]\s/gm, '<span style="color:#c9a96e;margin-right:4px">·</span>');
    answerDiv.innerHTML = answer;

    // 来源
    document.getElementById('faq-sources').innerHTML = d.faq_sources.length
      ? d.faq_sources.map(s=>`<div class="source-chip"><div class="src-meta"><span class="src-tag faq">官方</span>${s.section}</div>${s.content.substring(0,180)}</div>`).join('')
      : '<div class="source-chip" style="color:#9ca3af">暂无匹配的官方信息</div>';

    document.getElementById('review-sources').innerHTML = d.review_sources.length
      ? d.review_sources.map(s=>`<div class="source-chip">
          <div class="src-meta">
            <span class="src-tag ${s.sentiment==='好评'?'positive':s.sentiment==='差评'?'negative':'neutral'}">${s.sentiment}</span>
            ${s.review_date} · ${(s.room_type||'').substring(0,8)}
          </div>
          ${s.content.substring(0,180)}
        </div>`).join('')
      : '<div class="source-chip" style="color:#9ca3af">暂无匹配的住客评论</div>';

    sourcesDiv.style.display = 'grid';
    wrapper.scrollIntoView({behavior:'smooth',block:'start'});
  } catch(e) {
    answerDiv.innerHTML = '<span style="color:#991b1b">抱歉，服务暂时不可用：' + e.message + '</span>';
  }
}

function quickAsk(q) {
  document.getElementById('rag-question').value = q;
  askRAG();
}
</script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def index():
    return FRONTEND_HTML


# ============================================================
# 启动入口
# ============================================================

if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("🦢 白天鹅宾馆 · 评论智能分析系统")
    print("=" * 60)
    print(f"   SQLite:     {config.SQLITE_DB}")
    print(f"   Chroma:     {config.CHROMA_DIR}")
    print(f"   LLM:        {config.LLM_MODEL} ({'已配置' if config.LLM_API_KEY else '❌ 未配置'})")
    print(f"\n   启动服务: http://localhost:8000")
    print(f"   API文档:  http://localhost:8000/docs")
    print("=" * 60)
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
