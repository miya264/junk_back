from __future__ import annotations
import os
import re
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import asyncmy
from asyncmy.cursors import DictCursor
from asyncmy.errors import Error
from dotenv import load_dotenv, find_dotenv
import asyncio
import json

from core.config import chat, GBIZINFO_API_KEY, GBIZINFO_URL
from DB.mysql.mysql_connection import get_mysql_db, init_db_pool
from DB.mysql.mysql_crud import search_coworkers

router = APIRouter(prefix="/api/people", tags=["people"])

# Schemas
class SuggestBody(BaseModel):
    industry: List[str] = Field(default_factory=list)
    region:   List[str] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)
    limit: int = 12
    offset: int = 0
    chat_session_id: Optional[int] = None

class Candidate(BaseModel):
    card_id: int
    name: str
    company: str
    department: Optional[str] = None
    position: Optional[str] = None
    score: float
    reasons: List[str]

class SuggestResponse(BaseModel):
    items: List[Candidate]
    total: int

class AskInputResponse(BaseModel):
    needs_input: bool = True
    missing: List[str]
    examples: dict
    suggestions: Optional[dict] = None

class NetworkNode(BaseModel):
    id: str
    label: str
    kind: str

class NetworkEdge(BaseModel):
    source: str
    target: str
    label: str

class NetworkGraph(BaseModel):
    nodes: List[NetworkNode]
    edges: List[NetworkEdge]

class PeopleSearchRequest(BaseModel):
    query: str = Field(..., description="自然文の検索条件")
    top_k: int = Field(5, ge=1, le=50)
    coworker_id: int | None = Field(None)

class PeopleAskRequest(BaseModel):
    question: str
    top_k: int = 5
    coworker_id: int | None = None

class PeopleAskResponse(BaseModel):
    narrative: str
    queries: list[str]
    candidates: list[dict]

class CompanyInfo(BaseModel):
    id: int | None = None
    name: str
    corporate_number: str | None = None
    location: str | None = None
    company_type: str | None = None
    founding_date: str | None = None
    capital: str | None = None
    employee_number: int | None = None
    business_summary: str | None = None

class CompanyInfoResponse(BaseModel):
    company: CompanyInfo | None

# Helpers
def _dt_to_str(x):
    if isinstance(x, datetime):
        return x.isoformat()
    return x

def like_list(words: List[str]) -> List[str]:
    return [f"%{w}%" for w in words if w]

def or_like(col: str, n: int) -> str:
    if n <= 0:
        return "0"
    return "(" + " OR ".join([f"{col} LIKE %s" for _ in range(n)]) + ")"

def build_reasons(r_ok, k_ok, c_ok, a_ok) -> List[str]:
    out: List[str] = []
    if r_ok: out.append("地域一致")
    if k_ok: out.append(f"キーワード一致: {int(k_ok)}項目")
    if c_ok and c_ok > 0: out.append(f"過去コンタクト: 約{max(1, int(round(c_ok)))}件")
    if a_ok and a_ok > 0: out.append(f"同席履歴: 約{max(1, int(round(a_ok)))}回")
    return out or ["関連度スコア一致"]

def tokenize(q: str) -> List[str]:
    return [t for t in re.split(r"[^\wぁ-んァ-ン一-龥]+", (q or "").strip()) if t]

async def _execute_query_db2(query: str, params: tuple | None = None):
    db = get_mysql_db()
    return await db.execute_query(query, params)

@router.post("/suggest", response_model=SuggestResponse | AskInputResponse)
async def people_suggest(body: SuggestBody):
    if not body.region and not body.keywords:
        return AskInputResponse(
            missing=["region", "keywords"],
            examples={"region": ["富山県"], "keywords": ["ブランド", "EC"]},
            suggestions=None,
        )
    rg = like_list(body.region)
    kw = like_list(body.keywords)
    async with get_conn() as conn:
        cur = await conn.cursor(DictCursor)
        where = ["1=1"]
        params: List[object] = []
        if rg:
            where.append(or_like("c.location", len(rg)))
            params += rg
        if kw:
            where.append(
                "(" + " OR ".join([or_like("COALESCE(bc.company, c.name)", len(kw)), or_like("bc.department", len(kw)), or_like("bc.position", len(kw)), or_like("bc.memo", len(kw)),]) + ")"
            )
            params += kw + kw + kw + kw
        base_sql = f"""
          SELECT bc.id AS card_id, bc.name, COALESCE(bc.company, c.name) AS company, bc.department, bc.position
          FROM business_cards bc LEFT JOIN companies c ON c.id = bc.company_id
          WHERE {' AND '.join(where)}
        """
        await cur.execute(f"SELECT COUNT(1) AS cnt FROM ({base_sql}) t", params)
        total = (await cur.fetchone())["cnt"]
        score_sql = f"""
          WITH base AS ({base_sql}), kw AS (SELECT b.card_id, ( (CASE WHEN {" OR ".join([f"COALESCE(bc.company, c.name) LIKE %s" for _ in kw]) if kw else "0"} THEN 1 ELSE 0 END) + (CASE WHEN {" OR ".join([f"bc.department LIKE %s" for _ in kw]) if kw else "0"} THEN 1 ELSE 0 END) + (CASE WHEN {" OR ".join([f"bc.position LIKE %s" for _ in kw]) if kw else "0"} THEN 1 ELSE 0 END) + (CASE WHEN {" OR ".join([f"bc.memo LIKE %s" for _ in kw]) if kw else "0"} THEN 1 ELSE 0 END) ) AS k_score FROM base b JOIN business_cards bc ON bc.id = b.card_id LEFT JOIN companies c ON c.id = bc.company_id ), contact AS (SELECT b.card_id, IFNULL(LOG(1 + COUNT(DISTINCT ct.id)), 0) AS c_score FROM base b JOIN contact_person cp ON cp.business_card_id = b.card_id JOIN contacts ct ON ct.id = cp.contact_id GROUP BY b.card_id ), attend AS (SELECT b.card_id, IFNULL(LOG(1 + SUM(x.weight)), 0) AS a_score FROM base b LEFT JOIN (SELECT source_id AS aid, weight FROM co_attended_relations UNION ALL SELECT target_id AS aid, weight FROM co_attended_relations) x ON x.aid = b.card_id GROUP BY b.card_id) SELECT b.card_id, b.name, b.company, b.department, b.position, ( 1.0 * (CASE WHEN {" OR ".join([f"c.location LIKE %s" for _ in rg]) if rg else "0"} THEN 1 ELSE 0 END) + 0.8 * IFNULL(kw.k_score, 0) + 1.2 * IFNULL(contact.c_score, 0) + 1.0 * IFNULL(attend.a_score, 0) ) AS score, (CASE WHEN {" OR ".join([f"c.location LIKE %s" for _ in rg]) if rg else "0"} THEN 1 ELSE 0 END) AS r_ok, IFNULL(kw.k_score, 0) AS k_ok, IFNULL(contact.c_score, 0) AS c_ok, IFNULL(attend.a_score, 0) AS a_ok FROM base b LEFT JOIN business_cards bc ON bc.id = b.card_id LEFT JOIN companies c ON c.id = bc.company_id LEFT JOIN kw ON kw.card_id = b.card_id LEFT JOIN contact ON contact.card_id = b.card_id LEFT JOIN attend ON attend.card_id = b.card_id ORDER BY score DESC, b.card_id DESC LIMIT %s OFFSET %s
        """
        params2 = params.copy()
        if kw: params2 += kw * 4
        if rg: params2 += rg * 2
        params2 += [body.limit, body.offset]
        await cur.execute(score_sql, params2)
        rows = await cur.fetchall()
    items = [
        Candidate(card_id=r["card_id"], name=r["name"], company=r["company"] or "", department=r["department"], position=r["position"], score=float(r["score"]), reasons=build_reasons(r["r_ok"], r["k_ok"], r["c_ok"], r["a_ok"]),)
        for r in rows
    ]
    return SuggestResponse(items=items, total=total)

@router.get("/{card_id}/network", response_model=NetworkGraph)
async def get_network(card_id: int):
    nodes: List[NetworkNode] = []
    edges: List[NetworkEdge] = []
    async with get_conn() as conn:
        cur = await conn.cursor(DictCursor)
        await cur.execute(
            """
            SELECT bc.id, bc.name, COALESCE(bc.company, c.name) AS company
            FROM business_cards bc LEFT JOIN companies c ON c.id = bc.company_id
            WHERE bc.id = %s
            """, (card_id,),
        )
        row = await cur.fetchone()
        if not row: raise HTTPException(status_code=404, detail="business_card not found")
        center_id = f"card:{row['id']}"
        nodes.append(NetworkNode(id=center_id, label=f"{row['name']}", kind="中心"))
        await cur.execute(
            """
            SELECT cw.id AS coworker_id, cw.name AS coworker_name, d.name AS dept,
                   cr.first_contact_date, cr.last_contact_date, cr.contact_count
            FROM coworker_relations cr
            JOIN coworkers cw ON cw.id = cr.coworker_id
            LEFT JOIN departments d ON d.id = cw.department_id
            WHERE cr.business_card_id = %s
            ORDER BY COALESCE(cr.last_contact_date, cr.first_contact_date) DESC, cw.name
            """, (card_id,),
        )
        holders = await cur.fetchall()
    for h in holders:
        nid = f"cw:{h['coworker_id']}"
        label = f"{h['coworker_name']}" + (f"\n{h['dept']}" if h.get("dept") else "")
        nodes.append(NetworkNode(id=nid, label=label, kind="名刺保有者"))
        edges.append(NetworkEdge(source=center_id, target=nid, label="名刺保有者"))
    return NetworkGraph(nodes=nodes, edges=edges)

@router.post("/search", response_model=Dict[str, Any])
async def people_search_endpoint(req: PeopleSearchRequest):
    if not (req.query or "").strip():
        return {"candidates": []}
    eff_coworker_id = req.coworker_id if (req.coworker_id and req.coworker_id > 0) else None
    
    from langchain_openai import ChatOpenAI
    from langchain.prompts import PromptTemplate
    from langchain.schema import SystemMessage, HumanMessage
    from main import _people_generate_sql_with_llm, _people_sanitize_sql, _cleanup_nl_query
    
    async def _generate_and_sanitize_people_sql(nl_query: str, top_k: int, coworker_id: int | None):
        attempts = [
            dict(force_single_select=False, force_template=False),
            dict(force_single_select=True,  force_template=False),
            dict(force_single_select=True,  force_template=True),
        ]
        last_err = None
        for i, opt in enumerate(attempts, 1):
            try:
                raw_sql = await _people_generate_sql_with_llm(
                    nl_query, top_k, coworker_id,
                    force_single_select=opt["force_single_select"],
                    force_template=opt["force_template"],
                )
                safe_sql = _people_sanitize_sql(raw_sql, top_k)
                return safe_sql, raw_sql
            except HTTPException as e:
                last_err = e
                continue
        if last_err: raise last_err
        raise HTTPException(status_code=400, detail="SQL generation failed")
    
    safe_sql, raw_sql = await _generate_and_sanitize_people_sql(
        req.query.strip(), req.top_k, eff_coworker_id
    )

    db = get_mysql_db()
    try:
        rows = await db.execute_query(safe_sql)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB error: {e}")

    candidates = []
    for r in rows:
        candidates.append({
            "id": r.get("id"),
            "name": r.get("name"),
            "company": r.get("company") or "",
            "department": r.get("department"),
            "title": r.get("title"),
            "skills": None,
            "avatar_url": r.get("avatar_url"),
            "score": r.get("score") if r.get("score") is not None else 0,
        })
    if not candidates:
        results = await search_coworkers(q=req.query, department="")
        candidates = [{
            "id": r["id"], "name": r["name"], "company": "", "department": r.get("department_name"),
            "title": r.get("position"), "skills": None, "avatar_url": None, "score": 0,
        } for r in (results[:req.top_k] if isinstance(results, list) else [])]
    res = {"candidates": candidates}
    return UTF8JSONResponse(res)