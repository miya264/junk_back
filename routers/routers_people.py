# app/routers_people.py
from __future__ import annotations

"""
People Suggest & Network API (Azure MySQL)
- .env から DB_*2 を優先で読み込み（無ければ DB_*）
- SSL CA の相対パスを絶対化
- /people/suggest : 企画コンテキスト(JSON) → 候補人物（不足時は needs_input）
- /people/{card_id}/network : 名刺ID中心の関係図（coworker_relationsベース）
"""

import os
from contextlib import contextmanager
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

# ========== .env の読み込み ==========
try:
    from dotenv import load_dotenv, find_dotenv  # pip install python-dotenv
    env_path = find_dotenv(filename=".env", usecwd=True)
    load_dotenv(env_path, override=False)
except Exception:
    # dotenv が無くても OS 環境変数で動作可能
    pass


def getenv2(key2: str, key1: str, default: str | None = None) -> str | None:
    """DB_USER2 → DB_USER → default の優先で読む"""
    v = os.getenv(key2)
    if v is None or v == "":
        v = os.getenv(key1, default)
    return v


# ========== Azure MySQL 接続 (PyMySQL) ==========
import pymysql  # pip install PyMySQL
from pymysql.cursors import DictCursor

DB_USER = getenv2("DB_USER2", "DB_USER", "")
DB_PASS = getenv2("DB_PASS2", "DB_PASS", "")
DB_HOST = getenv2("DB_HOST2", "DB_HOST", "")
DB_NAME = getenv2("DB_NAME2", "DB_NAME", "")
DB_SSL_CA = getenv2("DB_SSL_CA2", "DB_SSL_CA", None)

# CA を絶対パスに補正（相対→絶対）
if DB_SSL_CA and not os.path.isabs(DB_SSL_CA):
    here = os.path.dirname(os.path.abspath(__file__))
    # プロジェクトルート = app/.. を想定
    DB_SSL_CA = os.path.abspath(os.path.join(here, "..", DB_SSL_CA))


@contextmanager
def get_conn():
    """
    Azure Database for MySQL への接続。
    うまく繋がらない場合は、ユーザー名を 'user@servername' 形式にすると通る環境があります。
    例: students@eastasiafor9th
    """
    kwargs = dict(
        host=DB_HOST,
        user=DB_USER or "",
        password=DB_PASS or "",
        database=DB_NAME or "",
        port=int(os.getenv("MYSQL_PORT", "3306")),
        charset="utf8mb4",
        cursorclass=DictCursor,
        autocommit=True,
    )
    if DB_SSL_CA:
        kwargs["ssl"] = {"ca": DB_SSL_CA}
    conn = pymysql.connect(**kwargs)
    try:
        yield conn
    finally:
        conn.close()


router = APIRouter(prefix="/people", tags=["people"])

# ========== Schemas ==========
class SuggestBody(BaseModel):
    industry: List[str] = Field(default_factory=list)   # 予備（将来拡張用）
    region:   List[str] = Field(default_factory=list)   # companies.location に LIKE
    keywords: List[str] = Field(default_factory=list)   # business_cards.* に LIKE
    limit: int = 12
    offset: int = 0
    chat_session_id: Optional[int] = None               # 予備：会話から推察など


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
    suggestions: Optional[dict] = None  # {"region":[...], "keywords":[...]}


class NetworkNode(BaseModel):
    id: str
    label: str
    kind: str  # '中心' | '名刺保有者'


class NetworkEdge(BaseModel):
    source: str
    target: str
    label: str


class NetworkGraph(BaseModel):
    nodes: List[NetworkNode]
    edges: List[NetworkEdge]


# ========== Helpers ==========
def like_list(words: List[str]) -> List[str]:
    return [f"%{w}%" for w in words if w]


def or_like(col: str, n: int) -> str:
    """(col LIKE %s OR col LIKE %s ... n回) を生成"""
    if n <= 0:
        return "0"  # 常に偽
    return "(" + " OR ".join([f"{col} LIKE %s" for _ in range(n)]) + ")"


def build_reasons(r_ok, k_ok, c_ok, a_ok) -> List[str]:
    out: List[str] = []
    if r_ok:
        out.append("地域一致")
    if k_ok:
        out.append(f"キーワード一致: {int(k_ok)}項目")
    if c_ok and c_ok > 0:
        out.append(f"過去コンタクト: 約{max(1, int(round(c_ok)))}件")
    if a_ok and a_ok > 0:
        out.append(f"同席履歴: 約{max(1, int(round(a_ok)))}回")
    return out or ["関連度スコア一致"]


# ========== Routes ==========
@router.post("/suggest", response_model=SuggestResponse | AskInputResponse)
def people_suggest(body: SuggestBody):
    """
    企画コンテキスト(JSON)を受け取り、人物候補を返す。
    入力（region/keywords）が空なら、不足検知レスポンスを返す。
    """
    # 1) 入力不足 → 聞き返し（サーバ側で判定）
    if not body.region and not body.keywords:
        return AskInputResponse(
            missing=["region", "keywords"],
            examples={
                "region": ["富山県", "北陸", "東京都"],
                "keywords": ["ブランド", "EC", "デザイン", "海外展示会"],
            },
            suggestions=None,  # 将来: chat_session_id を使って推察候補を返すなど
        )

    rg = like_list(body.region)
    kw = like_list(body.keywords)

    with get_conn() as conn:
        cur = conn.cursor()

        # 2) 母集合（location・各種フィールドで LIKE）
        where = ["1=1"]
        params: List[object] = []

        if rg:
            where.append(or_like("c.location", len(rg)))
            params += rg

        if kw:
            where.append(
                "("
                + " OR ".join(
                    [
                        or_like("COALESCE(bc.company, c.name)", len(kw)),
                        or_like("bc.department", len(kw)),
                        or_like("bc.position", len(kw)),
                        or_like("bc.memo", len(kw)),
                    ]
                )
                + ")"
            )
            params += kw + kw + kw + kw

        base_sql = f"""
          SELECT bc.id AS card_id,
                 bc.name,
                 COALESCE(bc.company, c.name) AS company,
                 bc.department,
                 bc.position
          FROM business_cards bc
          LEFT JOIN companies c ON c.id = bc.company_id
          WHERE {' AND '.join(where)}
        """

        # total 件数
        cur.execute(f"SELECT COUNT(1) AS cnt FROM ({base_sql}) t", params)
        total = cur.fetchone()["cnt"]

        # 3) スコア合成（MySQL 8 の CTE）
        #   r_score: 地域一致(0/1)
        #   k_score: キーワード一致の項目数（0..4）
        #   c_score: 過去コンタクト（log(1 + 件数)）
        #   a_score: 共起の重み（log(1 + sum(weight)））
        #   最終:  1.0*r + 0.8*k + 1.2*c + 1.0*a
        score_sql = f"""
          WITH base AS (
            {base_sql}
          ),
          kw AS (
            SELECT b.card_id,
              (
                (CASE WHEN {" OR ".join([f"COALESCE(bc.company, c.name) LIKE %s" for _ in kw]) if kw else "0"} THEN 1 ELSE 0 END) +
                (CASE WHEN {" OR ".join([f"bc.department LIKE %s"             for _ in kw]) if kw else "0"} THEN 1 ELSE 0 END) +
                (CASE WHEN {" OR ".join([f"bc.position LIKE %s"               for _ in kw]) if kw else "0"} THEN 1 ELSE 0 END) +
                (CASE WHEN {" OR ".join([f"bc.memo LIKE %s"                   for _ in kw]) if kw else "0"} THEN 1 ELSE 0 END)
              ) AS k_score
            FROM base b
            JOIN business_cards bc ON bc.id = b.card_id
            LEFT JOIN companies c ON c.id = bc.company_id
          ),
          contact AS (
            SELECT b.card_id, IFNULL(LOG(1 + COUNT(DISTINCT ct.id)), 0) AS c_score
            FROM base b
            JOIN contact_person cp ON cp.business_card_id = b.card_id
            JOIN contacts ct ON ct.id = cp.contact_id
            GROUP BY b.card_id
          ),
          attend AS (
            SELECT b.card_id, IFNULL(LOG(1 + SUM(x.weight)), 0) AS a_score
            FROM base b
            LEFT JOIN (
              SELECT source_id AS aid, weight FROM co_attended_relations
              UNION ALL
              SELECT target_id AS aid, weight FROM co_attended_relations
            ) x ON x.aid = b.card_id
            GROUP BY b.card_id
          )
          SELECT
            b.card_id, b.name, b.company, b.department, b.position,
            (
              1.0 * (CASE WHEN {" OR ".join([f"c.location LIKE %s" for _ in rg]) if rg else "0"} THEN 1 ELSE 0 END) +
              0.8 * IFNULL(kw.k_score, 0) +
              1.2 * IFNULL(contact.c_score, 0) +
              1.0 * IFNULL(attend.a_score, 0)
            ) AS score,
            (CASE WHEN {" OR ".join([f"c.location LIKE %s" for _ in rg]) if rg else "0"} THEN 1 ELSE 0 END) AS r_ok,
            IFNULL(kw.k_score, 0) AS k_ok,
            IFNULL(contact.c_score, 0) AS c_ok,
            IFNULL(attend.a_score, 0) AS a_ok
          FROM base b
          LEFT JOIN business_cards bc ON bc.id = b.card_id
          LEFT JOIN companies c ON c.id = bc.company_id
          LEFT JOIN kw ON kw.card_id = b.card_id
          LEFT JOIN contact ON contact.card_id = b.card_id
          LEFT JOIN attend  ON attend.card_id  = b.card_id
          ORDER BY score DESC, b.card_id DESC
          LIMIT %s OFFSET %s
        """

        # params2 は：
        # - base_sql と同じ where 用: params (rg + 4*kw)
        # - kw CTE 用: 追加で 4*kw
        # - 地域 CASE 用: 追加で 2*rg（score内と r_ok 用で2回）
        # - LIMIT/OFFSET: +2
        params2 = params.copy()
        if kw:
            params2 += kw * 4
        if rg:
            params2 += rg * 2
        params2 += [body.limit, body.offset]

        cur.execute(score_sql, params2)
        rows = cur.fetchall()

    items = [
        Candidate(
            card_id=r["card_id"],
            name=r["name"],
            company=r["company"] or "",
            department=r["department"],
            position=r["position"],
            score=float(r["score"]),
            reasons=build_reasons(r["r_ok"], r["k_ok"], r["c_ok"], r["a_ok"]),
        )
        for r in rows
    ]
    return SuggestResponse(items=items, total=total)


# ========== ネットワーク（coworker_relations ベース） ==========
@router.get("/{card_id}/network", response_model=NetworkGraph)
def get_network(card_id: int):
    """
    名刺（business_cards.id）を中心に、周囲に
    「その名刺を保有している社内メンバー（coworker_relations）」のみを配置して返す。
    """
    nodes: List[NetworkNode] = []
    edges: List[NetworkEdge] = []

    with get_conn() as conn:
        cur = conn.cursor()

        # 中心ノード（名刺）
        cur.execute(
            """
            SELECT bc.id, bc.name, COALESCE(bc.company, c.name) AS company
            FROM business_cards bc
            LEFT JOIN companies c ON c.id = bc.company_id
            WHERE bc.id = %s
            """,
            (card_id,),
        )
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="business_card not found")

        center_id = f"card:{row['id']}"
        nodes.append(NetworkNode(id=center_id, label=f"{row['name']}", kind="中心"))

        # 名刺を保有している社内メンバー（複数）
        cur.execute(
            """
            SELECT cw.id AS coworker_id, cw.name AS coworker_name, d.name AS dept,
                   cr.first_contact_date, cr.last_contact_date, cr.contact_count
            FROM coworker_relations cr
            JOIN coworkers cw       ON cw.id = cr.coworker_id
            LEFT JOIN departments d ON d.id = cw.department_id
            WHERE cr.business_card_id = %s
            ORDER BY COALESCE(cr.last_contact_date, cr.first_contact_date) DESC, cw.name
            """,
            (card_id,),
        )
        holders = cur.fetchall()

    # ノード＆エッジ整形（社内保有者のみ）
    for h in holders:
        nid = f"cw:{h['coworker_id']}"
        label = f"{h['coworker_name']}" + (f"\n{h['dept']}" if h.get("dept") else "")
        nodes.append(NetworkNode(id=nid, label=label, kind="名刺保有者"))
        edges.append(NetworkEdge(source=center_id, target=nid, label="名刺保有者"))

    return NetworkGraph(nodes=nodes, edges=edges)


# ========== デバッグ用（任意で残してOK） ==========
@router.get("/_debug/env")
def _debug_env():
    def mask(v: str | None):
        return (v[:2] + "****") if v else None

    return {
        "DB_USER_effective": DB_USER,
        "DB_HOST_effective": DB_HOST,
        "DB_NAME_effective": DB_NAME,
        "DB_PASS_effective": mask(DB_PASS),
        "DB_SSL_CA_exists": bool(DB_SSL_CA and os.path.exists(DB_SSL_CA)),
    }
