from fastapi import FastAPI, HTTPException, Response, Cookie, Depends
from fastapi.middleware.cors import CORSMiddleware
import asyncio
from concurrent.futures import ThreadPoolExecutor
from fastapi.responses import JSONResponse
import json
from pydantic import BaseModel
from typing import Optional, List, Dict
import os
import time
from functools import wraps
import hashlib
from dotenv import load_dotenv
from pathlib import Path
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.schema import SystemMessage, HumanMessage
from langchain_core.documents import Document
try:
    from pinecone import Pinecone
except Exception:
    Pinecone = None
import uuid
from datetime import datetime
try:
    from policy_agents import PolicyAgentSystem
except ImportError:
    PolicyAgentSystem = None

try:
    from flexible_policy_agents import FlexiblePolicyAgentSystem
except ImportError:
    FlexiblePolicyAgentSystem = None
import mysql.connector
from mysql.connector import Error

# 環境変数の読み込み（backend/.env を明示的に参照）
ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=False)

app = FastAPI(title="AI Agent API", version="1.0.0")

# 非同期処理用のスレッドプール
executor = ThreadPoolExecutor(max_workers=4)

# メモリキャッシュの実装
CACHE = {}
CACHE_TTL = 300  # 5分間のキャッシュ

def cache_key(*args, **kwargs):
    """キャッシュキーを生成"""
    key_string = str(args) + str(sorted(kwargs.items()))
    return hashlib.md5(key_string.encode()).hexdigest()

def memory_cache(ttl_seconds=CACHE_TTL):
    """メモリキャッシュデコレータ"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            key = f"{func.__name__}_{cache_key(*args, **kwargs)}"
            current_time = time.time()
            
            # キャッシュから取得
            if key in CACHE:
                cached_data, timestamp = CACHE[key]
                if current_time - timestamp < ttl_seconds:
                    return cached_data
                else:
                    del CACHE[key]  # 期限切れのキャッシュを削除
            
            # 新しいデータを取得してキャッシュ
            result = func(*args, **kwargs)
            CACHE[key] = (result, current_time)
            return result
        return wrapper
    return decorator

# UTF-8エンコーディングを明示的に設定
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

origins = [
    # ローカル開発環境のオリジン
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    # デプロイしたフロントエンドのオリジン
    "https://apps-junk-02.azurewebsites.net",
    # 必要に応じて他のオリジンを追加
]

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001", 
        "http://localhost:3002",
        "http://localhost:3003",
        "http://localhost:3004",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001",
        "http://127.0.0.1:3002", 
        "http://127.0.0.1:3003",
        "http://127.0.0.1:3004",
        "https://apps-junk-02.azurewebsites.net",
    ],
    allow_credentials=True,  # クッキー認証に必要
    allow_methods=["*"],
    allow_headers=["*"],
)

# カスタムJSONレスポンス（UTF-8エンコーディング保証）
class UTF8JSONResponse(JSONResponse):
    def __init__(self, content, **kwargs):
        kwargs.setdefault('media_type', 'application/json; charset=utf-8')
        super().__init__(content, **kwargs)

# 環境変数
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "rag-hakusho")

# モデルの初期化（環境変数が設定されている場合のみ）
embedding_model = None
chat = None
pc = None
index = None

if OPENAI_API_KEY:
    try:
        embedding_model = OpenAIEmbeddings(
            openai_api_key=OPENAI_API_KEY,
            model="text-embedding-3-small",
            chunk_size=1000,
        )
        
        chat = ChatOpenAI(
            openai_api_key=OPENAI_API_KEY,
            model="gpt-4o-mini",
            temperature=0.3,
        )
        print("✓ OpenAI models initialized successfully")
    except Exception as e:
        print(f"⚠️ Warning: OpenAI initialization failed: {e}")

if PINECONE_API_KEY and Pinecone:
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(INDEX_NAME)
        print("✓ Pinecone initialized successfully")
    except Exception as e:
        print(f"⚠️ Warning: Pinecone initialization failed: {e}")
else:
    print("⚠️ Warning: Pinecone not available")

if not OPENAI_API_KEY or not PINECONE_API_KEY:
    print("⚠️ Warning: Some AI services not available due to missing environment variables")
    print(f"   OPENAI_API_KEY: {'SET' if OPENAI_API_KEY else 'NOT SET'}")
    print(f"   PINECONE_API_KEY: {'SET' if PINECONE_API_KEY else 'NOT SET'}")
    print(f"   PINECONE_INDEX_NAME: {INDEX_NAME}")
    print("   📋 To fix this in Azure App Service:")
    print("   1. Go to Configuration → Application settings")
    print("   2. Add OPENAI_API_KEY with your OpenAI API key")  
    print("   3. Add PINECONE_API_KEY with your Pinecone API key")
    print("   4. Add PINECONE_INDEX_NAME with your index name (default: rag-hakusho)")
    print("   5. Restart the app service")

# 政策立案エージェントシステムの初期化（AI サービスが利用可能な場合のみ）
policy_system = None
flexible_policy_system = None

if chat and embedding_model and index and FlexiblePolicyAgentSystem:
    try:
        # インポートが必要
        try:
            from flexible_policy_agents import CrudSectionRepo, CrudChatRepo
        except ImportError:
            CrudSectionRepo = None
            CrudChatRepo = None
        
        # 既存のDB機能と統合するためのCRUDアダプター
        class MainCrud:
            def get_project_step_sections(self, project_id: str, step_key: str):
                try:
                    from DB.mysql_crud import get_project_step_sections
                    return get_project_step_sections(project_id, step_key)
                except:
                    return []
                    
            def get_recent_chat_messages(self, session_id: str, limit: int = 10):
                # チャット履歴機能は後で実装
                return []
        
        main_crud = MainCrud()
        if CrudSectionRepo and CrudChatRepo:
            section_repo = CrudSectionRepo(main_crud)
            chat_repo = CrudChatRepo(main_crud)
        else:
            section_repo = None
            chat_repo = None
        
        if PolicyAgentSystem:
            policy_system = PolicyAgentSystem(chat, embedding_model, index)
        
        if FlexiblePolicyAgentSystem and section_repo and chat_repo:
            flexible_policy_system = FlexiblePolicyAgentSystem(chat, section_repo, chat_repo)
        print("✓ Policy agent systems initialized successfully")
    except Exception as e:
        print(f"⚠️ Warning: Policy agent initialization failed: {e}")
else:
    print("⚠️ Warning: Policy agents not available - AI services not initialized")

# =====================
# MySQL helpers
# =====================
def _mysql_config():
    return {
        'host': os.getenv('DB_HOST'),
        'port': int(os.getenv('DB_PORT', 3306)),
        'user': os.getenv('DB_USER'),
        'password': os.getenv('DB_PASSWORD'),
        'database': os.getenv('DB_NAME'),
        'charset': 'utf8mb4',
    }

def _execute_query(query: str, params: tuple | None = None):
    # プールされた接続を使用（再利用によりオーバーヘッド削減）
    config = _mysql_config()
    config.update({
        "buffered": True,
    })
    # サポートされていないパラメータを削除
    config.pop('prepared', None)
    config.pop('pool_name', None)
    config.pop('pool_size', None)
    config.pop('pool_reset_session', None)
    
    conn = mysql.connector.connect(**config)
    cur = conn.cursor(dictionary=True, buffered=True)
    try:
        cur.execute(query, params or ())
        if query.strip().upper().startswith('SELECT'):
            return cur.fetchall()
        conn.commit()
        return []
    finally:
        try:
            cur.close()
        finally:
            conn.close()

# Pydanticモデル
class MessageRequest(BaseModel):
    content: str
    search_type: Optional[str] = "normal"
    flow_step: Optional[str] = None
    context: Optional[dict] = None
    session_id: Optional[str] = None
    project_id: Optional[str] = None

class PolicyStepRequest(BaseModel):
    content: str
    step: str
    context: Optional[dict] = None

class PolicyStepResponse(BaseModel):
    id: str
    content: str
    step: str
    timestamp: str
    context: Optional[dict] = None

class FlexiblePolicyResponse(BaseModel):
    id: str
    content: str
    step: str
    timestamp: str
    session_id: str
    project_id: Optional[str] = None
    navigate_to: Optional[str] = None  # ステップ移動のターゲット
    type: Optional[str] = None         # レスポンスタイプ（"navigate"等）
    full_state: Optional[dict] = None

class SessionStateResponse(BaseModel):
    session_id: str
    project_id: Optional[str] = None
    analysis_result: Optional[str] = None
    objective_result: Optional[str] = None
    concept_result: Optional[str] = None
    plan_result: Optional[str] = None
    proposal_result: Optional[str] = None
    last_updated_step: Optional[str] = None
    step_timestamps: Optional[dict] = None

class MessageResponse(BaseModel):
    id: str
    content: str
    type: str
    timestamp: str
    search_type: Optional[str] = None

class ChatSession(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str

class ProjectStepSectionRequest(BaseModel):
    project_id: str
    step_key: str
    sections: List[Dict[str, str]]  # [{"section_key": "problem", "content": "..."}, ...]

class ProjectStepSectionResponse(BaseModel):
    id: str
    project_id: str
    step_key: str
    section_key: str
    content: str
    created_at: str
    updated_at: str

class CoworkerResponse(BaseModel):
    id: int
    name: str
    position: Optional[str] = None
    email: str
    department_name: Optional[str] = None

class ProjectCreateRequest(BaseModel):
    name: str
    description: Optional[str] = None
    owner_coworker_id: int
    member_ids: List[int] = []

class ProjectResponse(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    status: str
    owner_coworker_id: int
    owner_name: str
    members: List[CoworkerResponse]
    created_at: str
    updated_at: str

# CRUD操作をインポート
from DB.mysql_crud import (
    search_coworkers,
    create_project,
    get_project_by_id,
    get_projects_by_coworker,
    save_project_step_sections,
    get_project_step_sections,
    health_check as db_health_check,
    CRUDError
)

# 認証関連のインポート
from auth import auth_service, LoginRequest, LoginResponse

# セッション管理（簡易版）
sessions = {}

# =====================
# DB utility functions
# =====================
def _get_step_id(project_id: Optional[str], step_key: Optional[str]) -> Optional[str]:
    if not project_id or not step_key:
        return None
    rows = _execute_query(
        "SELECT id FROM policy_steps WHERE project_id = %s AND step_key = %s",
        (project_id, step_key),
    )
    if rows:
        return rows[0]['id']
    # なければ作成
    new_id = str(uuid.uuid4())
    _execute_query(
        """
        INSERT INTO policy_steps (id, project_id, step_key, step_name, order_no, status)
        VALUES (%s, %s, %s, %s, 1, 'active')
        """,
        (new_id, project_id, step_key, step_key.title()),
    )
    return new_id

def _ensure_chat_session(session_id: str, project_id: Optional[str], step_key: Optional[str]) -> tuple[str, Optional[str]]:
    """chat_sessions に存在しなければ作成して session_id と step_id を返す"""
    # スキーマ上 project_id は NOT NULL なので、未指定ならDB保存はスキップ
    if not project_id:
        return session_id, None
    step_id = _get_step_id(project_id, step_key) if (project_id and step_key) else None
    _execute_query(
        """
        INSERT IGNORE INTO chat_sessions (id, project_id, step_id, title, created_by, created_at, updated_at)
        VALUES (%s, %s, %s, NULL, NULL, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        """,
        (session_id, project_id, step_id),
    )
    return session_id, step_id

def _save_chat_message(session_id: str, project_id: Optional[str], step_id: Optional[str], role: str, msg_type: str, content: str) -> None:
    # project_id が無い場合は保存スキップ（スキーマで NOT NULL）
    if not project_id:
        return
    _execute_query(
        """
        INSERT INTO chat_messages (id, session_id, project_id, step_id, role, msg_type, content, created_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
        """,
        (str(uuid.uuid4()), session_id, project_id, step_id, role, msg_type, content),
    )

def rerank_documents(query: str, docs: list[Document], chat: ChatOpenAI, top_k: int = 5) -> list[Document]:
    """文書の再ランキング"""
    prompt = PromptTemplate(
        input_variables=["query", "documents", "top_k"],
        template="""
あなたは優秀な政策アナリストです。以下はユーザーの質問と、その関連として検索された文書のリストです。
質問に最も関連性の高い上位{top_k}件の文書を選び、番号だけを出力してください。

質問: {query}

文書一覧:
{documents}

出力形式（文書番号のみ、例: 0,2,4）:
"""
)

    
    numbered_docs = []
    for i, doc in enumerate(docs):
        cleaned = doc.page_content[:200].replace("\n", " ")
        numbered_docs.append(f"{i}: {cleaned}")

    docs_text = "\n".join(numbered_docs)

    msg = HumanMessage(content=prompt.format(query=query, documents=docs_text, top_k=top_k))
    response = chat.invoke([msg])
    selected = []
    for s in response.content.split(","):
        try:
            idx = int(s.strip())
            if 0 <= idx < len(docs):
                selected.append(docs[idx])
        except:
            continue
    return selected

@memory_cache(ttl_seconds=600)  # 10分間キャッシュ
def perform_rag_search(query: str) -> tuple[str, list[dict]]:
    """RAG検索を実行し、回答テキストと出典リストを返す"""
    try:
        if not embedding_model or not index:
            return "申し訳ございませんが、現在RAG検索機能は利用できません。環境設定を確認してください。", []
        
        # タイムアウトを設定してembedding生成時間を制限
        query_embedding = embedding_model.embed_query(query)
        
        results = index.query(vector=query_embedding, top_k=15, include_metadata=True)

        matches = results.matches if hasattr(results, "matches") else results.get("matches", [])
        initial_docs = []
        for m in matches:
            meta  = m.metadata if hasattr(m, "metadata") else m.get("metadata", {}) or {}
            score = m.score    if hasattr(m, "score")    else m.get("score")
            text  = meta.get("text") or meta.get("chunk") or meta.get("page_content") or ""
            if not text:
                continue
            doc = Document(
                page_content=text,
                metadata={
                    "source": meta.get("source", ""),
                    "figure_section": meta.get("figure_section", ""),
                    "chunk_index": meta.get("chunk_index", ""),
                    "page_number": meta.get("page_number", ""),
                    "section_title": meta.get("section_title", ""),
                    "document_title": meta.get("document_title", ""),
                    "year": meta.get("year", ""),
                    "score": score,
                },
            )
            initial_docs.append(doc)


        # 再ランキング
        top_docs = rerank_documents(query, initial_docs, chat, top_k=5)

        # LLMに回答生成と出典情報の埋め込みを依頼
        documents_string = ""
        for i, doc in enumerate(top_docs, 1):
            # 出典名を構築
            source_name = doc.metadata.get('document_title', '不明')
            year = doc.metadata.get('year', '')
            section = doc.metadata.get('section_title', '')
            
            # 出典情報を「【文書名(年) - 章節】」の形式でまとめる
            source_info = f"【{source_name}"
            if year:
                source_info += f"（{year}年度）"
            if section:
                source_info += f" - {section}"
            source_info += "】"

            documents_string += f"{source_info}\n{doc.page_content.strip()}\n\n"

        prompt = PromptTemplate(
            template="""あなたは思考整理をサポートする壁打ち相手です。ユーザーからの質問と、それに関連する複数の参照資料が提供されます。
これらの資料を活用して、ユーザーへの応答を作成してください。

【基本ルール】
- 参照資料内の情報を要約して、自然な文章で回答してください。
- **必ず**本文中の該当する箇所に**提供された形式の出典情報（例：【〇〇白書 - 第3章】）を付与**してください。
- 参照資料に記載されていない推測や一般的な知識は含めないでください。
- 回答は、ユーザーの考えを引き出すような問いかけで締めくくってください。

【参照資料】
{documents}

【ユーザーからの質問】
{query}

【回答】
""",
            input_variables=["documents", "query"]
        )

        messages = [
            SystemMessage(content="あなたは思考整理をサポートする壁打ち相手です。回答の本文中に参照資料の情報を直接引用し、事実に基づいた応答を生成してください。"),
            HumanMessage(content=prompt.format(documents=documents_string, query=query))
        ]

        response = chat.invoke(messages)

        # 出典情報の配列を構築
        sources: list[dict] = []
        for doc in top_docs:
            sources.append({
                "source": doc.metadata.get("source"),
                "section_title": doc.metadata.get("section_title"),
                "document_title": doc.metadata.get("document_title"),
                "year": doc.metadata.get("year"),
                "page_number": doc.metadata.get("page_number"),
                "score": doc.metadata.get("score"),
            })

        return response.content.strip(), sources
        
    except Exception as e:
        return f"RAG検索中にエラーが発生しました: {str(e)}", []

def perform_policy_step(content: str, step: str, context: dict = None) -> str:
    """政策立案ステップ別処理"""
    try:
        if not policy_system:
            return "申し訳ございませんが、現在政策立案機能は利用できません。環境設定を確認してください。"
        
        result = policy_system.process_step(step, content, context)
        return result
    except Exception as e:
        return f"政策立案エージェント処理中にエラーが発生しました: {str(e)}"

def perform_normal_chat(content: str, session_id: str = None) -> str:
    """通常のチャット（自然な対話型壁打ち相手）"""
    try:
        if not chat:
            return "申し訳ございませんが、現在チャット機能は利用できません。環境設定を確認してください。"
        
        fact_context = ""
        if session_id and flexible_policy_system:
            try:
                session_state = flexible_policy_system._get_session_state(session_id)
                if session_state and session_state.get("fact_search_results"):
                    recent_facts = "\\n\\n".join(session_state["fact_search_results"][-2:])
                    fact_context = f"\\n\\n【これまでのファクト検索結果】\\n{recent_facts}"
            except Exception as e:
                print(f"Warning: Failed to get session state: {e}")

        user_input_with_context = content + fact_context if fact_context else content
        
        messages = [
            SystemMessage(content="""あなたは思考整理をサポートする壁打ち相手です。ユーザーとの自然な対話を通じて、以下の役割を果たしてください：
...（プロンプトは変更なし）...
"""),
            HumanMessage(content=user_input_with_context)
        ]
        
        response = chat.invoke(messages)
        return response.content.strip()
        
    except Exception as e:
        return f"チャット処理中にエラーが発生しました: {str(e)}"

@app.post("/api/chat", response_model=MessageResponse)
async def chat_endpoint(request: MessageRequest):
    """チャットエンドポイント"""
    try:
        # DB保存: ユーザーメッセージ
        sid = request.session_id or str(uuid.uuid4())
        session_id, step_id = _ensure_chat_session(sid, request.project_id, request.flow_step)
        _save_chat_message(session_id, request.project_id, step_id, 'user', request.search_type or 'normal', request.content)

        # 1. RAG検索ボタンが押された場合を最優先で処理
        if request.search_type == "fact":
            if not (embedding_model and index):
                ai_content = "申し訳ございませんが、現在RAG検索機能は利用できません。環境設定を確認してください。"
                sources = []
            else:
                ai_content, sources = perform_rag_search(request.content)
            
            if request.session_id and flexible_policy_system and hasattr(flexible_policy_system, "add_fact_search_result"):
                try:
                    flexible_policy_system.add_fact_search_result(request.session_id, ai_content)
                except Exception as e:
                    print(f"Warning: Failed to add fact search result: {e}")

            # RAG結果をDBに保存（検索クエリ＋出典）
            try:
                session_id, step_id = _ensure_chat_session(sid, request.project_id, request.flow_step)
                # rag_search_results は project_id, step_id が NOT NULL のため、揃っている時のみ保存
                if request.project_id and step_id:
                    _execute_query(
                        """
                        INSERT INTO rag_search_results
                          (id, project_id, step_id, session_id, query, result_text, result_json, sources_json, created_by, created_at)
                        VALUES
                          (%s, %s, %s, %s, %s, %s, NULL, %s, NULL, CURRENT_TIMESTAMP)
                        """,
                        (
                            str(uuid.uuid4()),
                            request.project_id,
                            step_id,
                            session_id,
                            request.content,
                            ai_content,
                            json.dumps(sources, ensure_ascii=False),
                        ),
                    )
            except Exception as e:
                print(f"WARN: failed to save rag_search_results: {e}")

        # 2. 政策立案ステップのボタンが押された場合
        elif request.flow_step:
            if not flexible_policy_system:
                ai_content = "申し訳ございませんが、現在政策立案機能は利用できません。環境設定を確認してください。"
            else:
                session_id = request.session_id or str(uuid.uuid4())
                project_id = request.project_id
                
                try:
                    result = flexible_policy_system.process_flexible(
                        request.content,
                        session_id,
                        request.flow_step,
                        project_id
                    )
                except Exception as e:
                    print(f"Error in flexible_policy_system.process_flexible: {e}")
                    result = {"error": "政策立案処理中にエラーが発生しました。"}
                
                if "error" in result:
                    ai_content = result["error"]
                else:
                    ai_content = result["result"]
            
        # 3. その他（人脈検索、通常のチャットなど）
        elif request.search_type == "network":
            if not chat:
                ai_content = "申し訳ございませんが、現在チャット機能は利用できません。環境設定を確認してください。"
            else:
                ai_content = perform_normal_chat(request.content)
        else:
            if not chat:
                ai_content = "申し訳ございませんが、現在チャット機能は利用できません。環境設定を確認してください。"
            else:
                ai_content = perform_normal_chat(request.content, request.session_id)
        
        ai_message = MessageResponse(
            id=str(uuid.uuid4()),
            content=ai_content,
            type="ai",
            timestamp=datetime.now().isoformat(),
            search_type=request.search_type
        )
        # DB保存: AIメッセージ
        _save_chat_message(session_id, request.project_id, step_id, 'ai', request.search_type or 'normal', ai_content)
        
        return UTF8JSONResponse(ai_message.dict())
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/policy-flexible", response_model=FlexiblePolicyResponse)
async def flexible_policy_endpoint(request: MessageRequest):
    """柔軟な政策立案エンドポイント"""
    try:
        if not request.flow_step:
            raise HTTPException(status_code=400, detail="flow_step is required")
        
        if not flexible_policy_system:
            raise HTTPException(status_code=503, detail="Policy system not available. Please check environment configuration.")
        
        session_id = request.session_id or str(uuid.uuid4())
        project_id = request.project_id
        
        try:
            result = flexible_policy_system.process_flexible(
                request.content,
                session_id,
                request.flow_step,
                project_id
            )
        except Exception as e:
            print(f"Error in flexible_policy_system.process_flexible: {e}")
            raise HTTPException(status_code=500, detail="政策立案処理中にエラーが発生しました。")
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        response = FlexiblePolicyResponse(
            id=str(uuid.uuid4()),
            content=result["result"],
            step=result["step"],
            timestamp=datetime.now().isoformat(),
            session_id=session_id,
            project_id=project_id,
            navigate_to=result.get("navigate_to"),  # ステップ移動情報
            type=result.get("type"),                # レスポンスタイプ
            full_state=result["full_state"]
        )

        # DB保存: セッション確保とメッセージ保存
        session_id, step_id = _ensure_chat_session(session_id, project_id, request.flow_step)
        _save_chat_message(session_id, project_id, step_id, 'user', 'normal', request.content)
        _save_chat_message(session_id, project_id, step_id, 'ai', 'normal', result["result"]) 
        
        return UTF8JSONResponse(response.dict())
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/session-state/{session_id}", response_model=SessionStateResponse)
async def get_session_state(session_id: str):
    """セッション状態を取得"""
    try:
        state = flexible_policy_system.get_session_state(session_id)
        
        if "error" in state:
            raise HTTPException(status_code=404, detail=state["error"])
        
        return SessionStateResponse(**state)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/sessions", response_model=List[ChatSession])
async def get_sessions():
    """セッション一覧を取得"""
    return list(sessions.values())

@app.post("/api/sessions", response_model=ChatSession)
async def create_session():
    """新しいセッションを作成"""
    session_id = str(uuid.uuid4())
    now = datetime.now()
    
    session = ChatSession(
        id=session_id,
        title=f"新しいチャット {now.strftime('%Y-%m-%d %H:%M')}",
        created_at=now.isoformat(),
        updated_at=now.isoformat()
    )
    
    sessions[session_id] = session
    return session

@app.post("/api/project-step-sections", response_model=List[ProjectStepSectionResponse])
async def save_step_sections(request: ProjectStepSectionRequest):
    """プロジェクトステップセクションを保存"""
    try:
        print(f"DEBUG: Received request: project_id={request.project_id}, step_key={request.step_key}, sections_count={len(request.sections)}")
        print(f"DEBUG: Request sections: {request.sections}")
        
        saved_sections = save_project_step_sections(request.project_id, request.step_key, request.sections)
        
        # セクション保存後にキャッシュを無効化
        try:
            from DB.mysql_crud import invalidate_project_cache
            invalidate_project_cache(request.project_id)
        except Exception as e:
            print(f"Warning: Cache invalidation failed: {e}")
        
        print(f"DEBUG: Successfully saved {len(saved_sections)} sections")
        return UTF8JSONResponse([ProjectStepSectionResponse(**section).dict() for section in saved_sections])
    except CRUDError as e:
        print(f"DEBUG: CRUDError occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        print(f"DEBUG: Exception occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/project-step-sections/{project_id}/{step_key}", response_model=List[ProjectStepSectionResponse])
async def get_step_sections(project_id: str, step_key: str):
    """プロジェクトステップセクションを取得"""
    try:
        sections = get_project_step_sections(project_id, step_key)
        return UTF8JSONResponse([ProjectStepSectionResponse(**section).dict() for section in sections])
    except CRUDError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/project-all-sections/{project_id}")
async def get_project_all_sections_endpoint(project_id: str):
    """プロジェクトの全ステップセクションを一括取得（高速化）"""
    try:
        from DB.mysql_crud import MySQLCRUD
        crud = MySQLCRUD()
        sections_by_step = crud.get_project_all_step_sections(project_id)
        return UTF8JSONResponse(sections_by_step)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/coworkers/search", response_model=List[CoworkerResponse])
async def search_coworkers_endpoint(q: str = "", department: str = ""):
    """coworkers検索"""
    try:
        coworkers = search_coworkers(q, department)
        return UTF8JSONResponse([CoworkerResponse(**coworker).dict() for coworker in coworkers])
    except CRUDError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/projects", response_model=ProjectResponse)
async def create_project_endpoint(request: ProjectCreateRequest):
    """プロジェクト作成"""
    try:
        project = create_project(
            request.name, 
            request.description or "", 
            request.owner_coworker_id, 
            request.member_ids
        )
        if not project:
            raise HTTPException(status_code=400, detail="Failed to create project")
        
        return UTF8JSONResponse(ProjectResponse(**project).dict())
    except CRUDError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/projects/{project_id}", response_model=ProjectResponse)
async def get_project_endpoint(project_id: str):
    """プロジェクト詳細取得"""
    try:
        project = get_project_by_id(project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        return UTF8JSONResponse(ProjectResponse(**project).dict())
    except CRUDError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/projects/by-coworker/{coworker_id}", response_model=List[ProjectResponse])
async def get_projects_by_coworker_endpoint(coworker_id: int):
    """coworkerが参加しているプロジェクト一覧取得"""
    try:
        projects = get_projects_by_coworker(coworker_id)
        return UTF8JSONResponse([ProjectResponse(**project).dict() for project in projects])
    except CRUDError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =====================
# 認証エンドポイント
# =====================

@app.post("/api/auth/login", response_model=LoginResponse)
async def login_endpoint(request: LoginRequest, response: Response):
    """ログインエンドポイント"""
    try:
        return await auth_service.login(request, response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/auth/logout")
async def logout_endpoint(response: Response):
    """ログアウトエンドポイント"""
    try:
        await auth_service.logout(response)
        return {"message": "Successfully logged out"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/auth/me")
async def get_current_user_endpoint(access_token: str = Cookie(None)):
    """現在のログインユーザー情報を取得"""
    try:
        user = await auth_service.get_current_user(access_token)
        if not user:
            raise HTTPException(status_code=401, detail="認証が必要です")
        return user
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/auth/verify")
async def verify_token_endpoint(access_token: str = Cookie(None)):
    """トークンの有効性を確認"""
    try:
        user = await auth_service.get_current_user(access_token)
        if not user:
            return {"valid": False}
        return {"valid": True, "user": user}
    except Exception as e:
        return {"valid": False, "error": str(e)}

@app.get("/")
async def root():
    """ルートエンドポイント"""
    return {"message": "AI Agent API is running"}

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)


# 旧 /api/me は削除（認証ベースの /api/auth/me に統合済み）