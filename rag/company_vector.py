# company_vector.py
import os
from typing import List, Dict, Any
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone

# .env をロード
load_dotenv()

# ========= 環境変数 =========
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# 会社検索用の専用インデックス名（未設定なら company-info）
PINECONE_COMPANY_INDEX_NAME = os.getenv("PINECONE_COMPANY_INDEX_NAME", "company-info")

# 既存 main.py の DB2 ヘルパと同じ仕様で import される想定
# ここでは関数参照だけにしておき、main.py から渡してもらう
# def _execute_query_db2(query: str, params: tuple | None = None): ...

# ========= 初期化 =========
pc = None
index = None
embedding_model = None

def _init_clients():
    global pc, index, embedding_model
    if not pc:
        pc = Pinecone(api_key=PINECONE_API_KEY)
    if not index:
        index = pc.Index(PINECONE_COMPANY_INDEX_NAME)
    if not embedding_model:
        # 既存RAGと同じモデルでOK（インデックスのdimensionはこのモデルに合わせて作成してください）
        embedding_model = OpenAIEmbeddings(
            openai_api_key=OPENAI_API_KEY,
            model="text-embedding-3-small"
        )

# ========= ユーティリティ =========

def _to_meta_value(v):
    """Pinecone metadataは None を許容しないので安全化"""
    if v is None:
        return ""
    if isinstance(v, (str, int, float, bool)):
        return v
    return str(v)

def _company_row_to_metadata(row: Dict[str, Any]) -> Dict[str, Any]:
    # 必要なフィールドだけ詰める（値は全部 safe 化）
    keep = {
        "id", "name", "corporate_number", "postal_code", "location",
        "company_type", "founding_date", "capital", "employee_number",
        "business_summary", "update_date"
    }
    meta = {}
    for k in keep:
        meta[k] = _to_meta_value(row.get(k))
    return meta

# ========= 再インデックス =========

def reindex_from_mysql(execute_query_db2) -> int:
    """
    junk_db.companies から全件読み出し、Pinecone に upsert します。
    戻り値: upsert 件数
    """
    _init_clients()
    rows = execute_query_db2("""
        SELECT id, name, corporate_number, postal_code, location, company_type,
               founding_date, capital, employee_number, business_summary, update_date
        FROM companies
        ORDER BY id ASC
    """)
    vectors = []
    for r in rows:
        # 検索に効かせたいテキストをまとめる
        text = " ".join([
            str(r.get("name") or ""),
            str(r.get("location") or ""),
            str(r.get("business_summary") or ""),
            str(r.get("company_type") or ""),
        ]).strip()

        # ベクトル化
        vec = embedding_model.embed_query(text)

        # id は文字列で一意に（corp num があれば corp、無ければ DB id）
        vid = str(r.get("corporate_number") or f"company:{r.get('id')}")
        vectors.append({
            "id": vid,
            "values": vec,
            "metadata": _company_row_to_metadata(r)
        })

    if vectors:
        # バッチ分割（念のため 100ずつ）
        batch = 100
        for i in range(0, len(vectors), batch):
            index.upsert(vectors=vectors[i:i+batch])
    return len(vectors)

# ========= 検索 =========

def search_companies(query: str, top_k: int = 10) -> List[Dict[str, Any]]:
    """
    ベクトル検索で会社候補を返す（name/location/summary での類似検索）
    戻り値: [{id, name, location, corporate_number, score}, ...]
    """
    _init_clients()
    qv = embedding_model.embed_query(query)
    res = index.query(vector=qv, top_k=top_k, include_metadata=True)
    items = []
    for m in getattr(res, "matches", []):
        md = m.metadata or {}
        items.append({
            "vector_id": m.id,
            "score": float(getattr(m, "score", 0.0) or 0.0),
            "id": md.get("id"),  # MySQLのcompanies.id
            "name": md.get("name"),
            "corporate_number": md.get("corporate_number"),
            "location": md.get("location"),
            "company_type": md.get("company_type"),
            "business_summary": md.get("business_summary"),
        })
    return items
