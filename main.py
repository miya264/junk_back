from fastapi import FastAPI, HTTPException, Response, Cookie, Depends
from fastapi.middleware.cors import CORSMiddleware
import asyncio
from concurrent.futures import ThreadPoolExecutor
from fastapi.responses import JSONResponse
import json
from pydantic import BaseModel
from pydantic import Field  # 再import可
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
import re as _re
import json as _json 

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

# CORS設定 - ローカル開発とAzure両対応
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        # ローカル開発
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
        # Azure デプロイ環境（実際のURL）
        "https://apps-junk-02.azurewebsites.net",
        "https://aps-junk-01-fbgncnexhuekadft.canadacentral-01.azurewebsites.net",
        "https://apps-junk-01.azurewebsites.net",
    ],
    allow_origin_regex=r"https://.*\.azurewebsites\.net",  # Azure動的URLパターン
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

@app.get("/")
async def root():
    """ルートエンドポイント"""
    return {"message": "AI Agent API is running"}

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

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
    
# =====================
# プロジェクトエンドポイント
# =====================

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
# セッションエンドポイント
# =====================

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
    
# =====================
# チャットエンドポイント
# =====================

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
    
# =====================
# チャット（エージェント）エンドポイント
# =====================

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
    
# =====================
# チャット（RAG）エンドポイント
# =====================

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
    

# =====================
# 人脈検索エンドポイント
# =====================

# LLM に使わせてよいテーブルとカラムを明示（これ以外は不可）
ALLOWED_TABLES = {
    "business_cards": [
        "id", "name", "company", "department", "position", "memo",
        "owner_coworker_id", "corporate_number", "company_id"
    ],
    "companies": [
        "id", "name", "corporate_number", "postal_code", "location",
        "company_type", "founding_date", "capital", "employee_number",
        "business_summary", "update_date"
    ],
    "coworker_relations": [
        "coworker_id", "business_card_id", "first_contact_date",
        "last_contact_date", "contact_count"
    ],
}

# 役に立たない汎用語（WHEREに入ると0件になりやすい語を拡充）
GENERIC_TOKENS: set[str] = {
    "社員", "担当者", "従業員", "役職", "職種", "部門", "部署",
    "会社", "企業", "日本", "国内", "海外", "本社",
    # 業種・業界系（bc.department に誤って入れがち）
    "業種", "業界", "小売", "小売業", "サービス業", "メーカー", "製造業", "IT業界"
}

def _cleanup_nl_query(q: str) -> str:
    """
    自然文のキーワードを正規化:
    - / ・、・,・空白で分割（全角空白も半角に）
    - 汎用語/業種語を除去
    - 重複排除（順序は維持）
    例: '任天堂/ゲーム/日本/役職/部門/社員' -> '任天堂 ゲーム'
    """
    q = (q or "").replace("　", " ")  # 全角空白→半角
    q = q.replace("／", "/").replace("、", "/").replace(",", "/")
    parts = [p.strip() for p in re.split(r"[\/\s]+", q) if p.strip()]
    parts = [p for p in parts if p not in GENERIC_TOKENS]
    seen: set[str] = set()
    cleaned: list[str] = []
    for p in parts:
        if p not in seen:
            seen.add(p)
            cleaned.append(p)
    return " ".join(cleaned) if cleaned else (q.strip() or "")



# --- DB2 (junk_db) 向けの接続ヘルパ（人物検索専用） ---
def _mysql_config_db2():
    return {
        'host': os.getenv('DB_HOST2'),
        'port': int(os.getenv('DB_PORT2', 3306)),
        'user': os.getenv('DB_USER2'),
        'password': os.getenv('DB_PASSWORD2'),
        'database': os.getenv('DB_NAME2'),
        'ssl_ca': os.getenv('DB_SSL_CA_PATH2'),
        'charset': 'utf8mb4',
        'autocommit': True,
    }

def _execute_query_db2(query: str, params: tuple | None = None):
    cfg = _mysql_config_db2()
    connect_kwargs = dict(
        host=cfg['host'],
        port=cfg['port'],
        user=cfg['user'],
        password=cfg['password'],
        database=cfg['database'],
        charset=cfg.get('charset', 'utf8mb4'),
        autocommit=cfg.get('autocommit', True),
        use_pure=True,
    )
    if cfg.get('ssl_ca'):
        connect_kwargs['ssl_ca'] = cfg['ssl_ca']
    conn = mysql.connector.connect(**connect_kwargs)
    cur = conn.cursor(dictionary=True)
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

# --- ネットワーク図データを作る（中心=名刺、外周=名刺保有者） -----------------
def _get_network_for_card(card_id: int) -> dict:
    # 中心（名刺）
    sql_center = """
        SELECT bc.id, bc.name, COALESCE(bc.company, c.name) AS company
        FROM business_cards bc
        LEFT JOIN companies c ON c.id = bc.company_id
        WHERE bc.id = %s
    """
    rows = _execute_query_db2(sql_center, (card_id,))
    if not rows:
        raise HTTPException(status_code=404, detail="business_card not found")
    center = rows[0]
    center_id = f"card:{center['id']}"
    nodes = [{"id": center_id, "label": f"{center['name']}", "kind": "中心"}]
    edges = []

    # 名刺を保有している社内メンバー
    sql_holders = """
        SELECT cw.id AS coworker_id, cw.name AS coworker_name, d.name AS dept,
               cr.first_contact_date, cr.last_contact_date, cr.contact_count
        FROM coworker_relations cr
        JOIN coworkers cw       ON cw.id = cr.coworker_id
        LEFT JOIN departments d ON d.id = cw.department_id
        WHERE cr.business_card_id = %s
        ORDER BY COALESCE(cr.last_contact_date, cr.first_contact_date) DESC, cw.name
    """
    holders = _execute_query_db2(sql_holders, (card_id,))
    for h in holders:
        nid = f"cw:{h['coworker_id']}"
        label = f"{h['coworker_name']}" + (f"\n{h['dept']}" if h.get("dept") else "")
        nodes.append({"id": nid, "label": label, "kind": "名刺保有者"})
        edges.append({"source": center_id, "target": nid, "label": "名刺保有者"})

    return {"nodes": nodes, "edges": edges}


# デバッグ出力（生成SQLを返す）
PEOPLE_SQL_DEBUG  = os.getenv("PEOPLE_SQL_DEBUG", "0") == "1"

class PeopleSearchRequest(BaseModel):
    query: str = Field(..., description="自然文の検索条件（例：'富山 EC デザイン'）")
    top_k: int = Field(5, ge=1, le=50, description="最大件数")
    coworker_id: int | None = Field(None, description="優先したい同僚のID（任意）")

def _people_generate_sql_with_llm(
    nl_query: str,
    top_k: int,
    coworker_id: int | None,
    force_single_select: bool = False,
    force_template: bool = False,
) -> str:
    """
    日本語→MySQL SELECT を1文だけ生成。必要に応じて JOIN。
    force_single_select: 「SELECT は1回だけ」と強く制約
    force_template: SELECT … FROM business_cards bc … LIMIT n の形を強制
    """
    if not chat:
        raise HTTPException(status_code=500, detail="OpenAI (chat) が初期化されていません")

    # スキーマをプロンプトへ
    schema_desc = []
    for tbl, cols in ALLOWED_TABLES.items():
        schema_desc.append(f"- {tbl}({', '.join(cols)})")
    schema_text = "\n".join(schema_desc)

    rules = [
        # ← 明確に「SELECT は1回だけ」を宣言（単語レベル）
        "Generate EXACTLY ONE SELECT statement only. The word SELECT must appear EXACTLY ONCE.",
        # ← サブクエリ全面禁止を “(SELECT …)” まで名指しで明示
        "NO subqueries of any kind: NO EXISTS(...), NO IN (SELECT ...), NO (SELECT ...) in any expression, NO CTEs, NO UNION.",
        "The main table MUST be business_cards (alias bc).",
        "Use LEFT JOIN companies c ON c.id = bc.company_id when needed.",
        "Use LEFT JOIN coworker_relations cr ON cr.business_card_id = bc.id when needed.",
        "Use only allowed columns. Return these aliases: id, name, company, department, title, avatar_url, score",
        "company := COALESCE(c.name, bc.company), title := COALESCE(bc.title, bc.position)",
        "ALWAYS ORDER BY score DESC, id ASC.",
        "ALWAYS include LIMIT (the number will be overwritten later).",
        'Output MUST be a single JSON object: {"sql": "..."} . No extra text, no code fences.',
        # 文字列比較は常に部分一致で
        "All text filters MUST use LIKE with wildcards: LIKE CONCAT('%', <keyword>, '%'). "
        "Never use equality (=) for company names or titles.",
        # 会社名のゆらぎ対策（“株式会社”“(株)”や空白を無視）
        "When matching company names, normalize by stripping '株式会社', '(株)', and spaces using "
        "REPLACE(REPLACE(REPLACE(COALESCE(c.name, bc.company),'株式会社',''),'(株)',''),' ',''). "
        "Compare that normalized value with LIKE CONCAT('%', <company_keyword>, '%').",
            # ★ ここを新規追加：業種語は無視する指示
        "Industry/sector words (e.g., 小売業, 製造業, IT業界) MUST NOT be mapped to bc.department; "
        "bc.department is an internal team like 営業部/開発部. If the user mentions an industry, ignore that condition."
    ]
    if force_single_select:
        rules.insert(0, "The word SELECT must appear EXACTLY ONCE in the whole statement.")
    if force_template:
        rules.insert(0, "Your SQL MUST match this shape strictly: "
                        "SELECT <columns> FROM business_cards bc"
                        "[ LEFT JOIN <table> <alias> ON <cond> ]*"
                        "[ WHERE <conditions> ]* ORDER BY <expr> LIMIT <n>")

    system = (
        "You are a strict SQL generator for MySQL 8.\n"
        f"Allowed tables and columns:\n{schema_text}\n\n"
        "Rules:\n- " + "\n- ".join(rules) + "\n"
    )

    hints = []
    if coworker_id is not None:
        hints.append(
            f"Prefer or filter by coworker_relations.coworker_id = {coworker_id} "
            "(e.g., add COALESCE(cr.contact_count,0) to score)."
        )
    hints_text = ("\nHints:\n- " + "\n- ".join(hints)) if hints else ""

    example = (
        "{\n"
        "  \"sql\": \"SELECT bc.id AS id, bc.name AS name,\n"
        "                  COALESCE(c.name, bc.company) AS company,\n"
        "                  bc.department AS department,\n"
        "                  COALESCE(bc.title, bc.position) AS title,\n"
        "                  bc.avatar_url AS avatar_url,\n"
        "                  (COALESCE(cr.contact_count,0)) AS score\n"
        "           FROM business_cards bc\n"
        "           LEFT JOIN companies c ON c.id = bc.company_id\n"
        "           LEFT JOIN coworker_relations cr ON cr.business_card_id = bc.id\n"
        "           ORDER BY score DESC, id ASC LIMIT 5\"\n"
        "}"
    )

    user = (
        "User intent (Japanese natural language):\n"
        f"{nl_query}\n\n"
        f"{hints_text}\n"
        "Write the query WITHOUT any subqueries; if you think you need EXISTS/IN, "
        "always rewrite it using LEFT JOIN + WHERE or JOIN-based scoring.\n"
        "Return JSON only, for example:\n" + example
    )

    resp = chat.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    content_text = str(resp.content).strip()

    m = _re.search(r"\{[\s\S]*\}", content_text)
    if not m:
        raise HTTPException(status_code=500, detail=f"LLM出力の解析に失敗しました: {content_text[:200]}")
    try:
        data = json.loads(m.group(0))
        sql = (data.get("sql") or "").strip()
        if not sql:
            raise ValueError("empty sql")
        return sql
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM JSON 解析に失敗: {e}")

_ALLOWED_TABLE_NAMES = set(ALLOWED_TABLES.keys())

def _people_sanitize_sql(sql: str, top_k: int) -> str:
    """
    - 先に存在しない列を安全な列へ書き換え
    - その後、DML/DDL/UNION/CTE/サブクエリ禁止 などを検査
    - LIMIT を強制
    """
    s = sql.strip().rstrip(";")

    # ★★★ ここを追加：存在しない列を強制リライト（大文字小文字を無視）
    s = _re.sub(r"\bbc\.title\b", "bc.position", s, flags=_re.I)
    s = _re.sub(r"\bbc\.avatar_url\b", "NULL",        s, flags=_re.I)

    # 以降は今のサニタイズのまま（Only single SELECT, (select…)/exists/in(select) 禁止 など）
    low_raw = s.lower()
    banned_regexes = [
        r";", r"\binsert\b", r"\bupdate\b", r"\bdelete\b",
        r"\bdrop\b", r"\balter\b", r"\bcreate\b", r"\bgrant\b", r"\brevoke\b", r"\btruncate\b",
        r"\bunion\b", r"\bwith\b",
        r"\bin\s*\(\s*select\b", r"\bexists\s*\(", r"\(\s*select\b",
    ]
    for pat in banned_regexes:
        if _re.search(pat, low_raw):
            raise HTTPException(status_code=400, detail="Unsafe SQL is not allowed")

    low = _re.sub(r"\s+", " ", low_raw)

    num_selects = len(_re.findall(r"\bselect\b", low_raw, flags=_re.I))
    if num_selects != 1:
        raise HTTPException(status_code=400, detail="Only a single SELECT is allowed")

    if " from business_cards" not in low:
        raise HTTPException(status_code=400, detail="Root table must be business_cards")

    tables = _re.findall(r"\b(from|join)\s+([a-zA-Z0-9_]+)", low)
    used_tables = {t[1] for t in tables}
    if not used_tables.issubset(_ALLOWED_TABLE_NAMES):
        bad = sorted(list(used_tables - _ALLOWED_TABLE_NAMES))
        raise HTTPException(status_code=400, detail=f"Disallowed tables detected: {bad}")

    if _re.search(r"\blimit\b\s+\d+", s, flags=_re.I):
        s = _re.sub(r"\blimit\b\s+\d+", f"LIMIT {int(top_k)}", s, flags=_re.I)
    else:
        s = s + f" LIMIT {int(top_k)}"

    return s


# ★ ここにリトライ用ヘルパを追加
def _generate_and_sanitize_people_sql(nl_query: str, top_k: int, coworker_id: int | None):
    """
    1回目: 通常生成 → サニタイズ
    2回目: 「SELECTは1回だけ」を強制して再生成 → サニタイズ
    3回目: さらにテンプレート強制で再生成 → サニタイズ
    いずれか成功した段階で返す。すべて失敗なら最後のエラーを送出。
    """
    attempts = [
        dict(force_single_select=False, force_template=False),
        dict(force_single_select=True,  force_template=False),
        dict(force_single_select=True,  force_template=True),
    ]

    last_err = None
    for i, opt in enumerate(attempts, 1):
        try:
            raw_sql = _people_generate_sql_with_llm(
                nl_query, top_k, coworker_id,
                force_single_select=opt["force_single_select"],
                force_template=opt["force_template"],
            )
            safe_sql = _people_sanitize_sql(raw_sql, top_k)
            if PEOPLE_SQL_DEBUG:
                print(f"[people-sql attempt {i}] OK\nRAW={raw_sql}\nSAFE={safe_sql}")
            return safe_sql, raw_sql
        except HTTPException as e:
            last_err = e
            # デバッグ用に生SQLも出す（サニタイズ前に落ちるケースはraw_sqlが無いので無視）
            try:
                print(f"[people-sql attempt {i}] FAIL: {e.detail}  (raw maybe above)")
            except Exception:
                pass
            continue

    # 3回とも弾かれた
    if last_err:
        raise last_err
    raise HTTPException(status_code=400, detail="SQL generation failed")

@app.post("/api/people/search")
async def people_search_endpoint(req: PeopleSearchRequest):
    if not (req.query or "").strip():
        return UTF8JSONResponse({"candidates": []})

    eff_coworker_id = req.coworker_id if (req.coworker_id and req.coworker_id > 0) else None

    safe_sql, raw_sql = _generate_and_sanitize_people_sql(
        req.query.strip(), req.top_k, eff_coworker_id
    )

    # 実行（DB2: junk_db）
    try:
        rows = _execute_query_db2(safe_sql, None)
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

    # 0件なら coworkers 検索にフォールバック
    if not candidates:
        try:
            results = search_coworkers(q=req.query, department="")
            candidates = [{
                "id":         r["id"],
                "name":       r["name"],
                "company":    "",
                "department": r.get("department_name"),
                "title":      r.get("position"),
                "skills":     None,
                "avatar_url": None,
                "score":      0,
            } for r in (results[:req.top_k] if isinstance(results, list) else [])]
        except Exception as ex:
            raise HTTPException(status_code=500, detail=f"coworkers 検索のフォールバックに失敗しました: {ex}") from ex

    res = {"candidates": candidates}
    if PEOPLE_SQL_DEBUG:
        res["debug_sql"] = {"raw": raw_sql, "sanitized": safe_sql}
    return UTF8JSONResponse(res)

# =====================
# People Search (LLM→SQL with JOINs) ここまで
# =====================

# =====================
# (追加) LLMファーストの人物探索 /api/people/ask
# =====================
from pydantic import BaseModel

class PeopleAskRequest(BaseModel):
    """自然文の質問から、LLMが検索クエリを立てて人物候補を返す"""
    question: str
    top_k: int = 5
    coworker_id: int | None = None  # 任意: 自分(や同僚)IDを優先度のヒントに使う

class PeopleAskResponse(BaseModel):
    narrative: str                     # LLMが返す前置きテキスト（「こういう人に当たりましょう」など）
    queries: list[str]                 # LLMが組み立てた自然文クエリ（/api/people/search にそのまま渡せる）
    candidates: list[dict]             # 人物カード（id, name, company, department, title, score など）
    # debug: dict | None = None       # 必要ならデバッグも返せます

def _people_search_core(nl_query: str, top_k: int, coworker_id: int | None) -> tuple[list[dict], dict]:
    """
    既存の人物検索ロジックを関数化（/api/people/search と同等）。
    返り値: (candidates, debug_sql)
    """
    eff_coworker_id = coworker_id if (coworker_id and coworker_id > 0) else None
    safe_sql, raw_sql = _generate_and_sanitize_people_sql(nl_query.strip(), top_k, eff_coworker_id)

    try:
        rows = _execute_query_db2(safe_sql, None)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB error: {e}")

    candidates: list[dict] = []
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

    # 0件なら coworkers テーブルにフォールバック（既存と同じ）
    if not candidates:
        try:
            results = search_coworkers(q=nl_query, department="")
            candidates = [{
                "id":         r["id"],
                "name":       r["name"],
                "company":    "",
                "department": r.get("department_name"),
                "title":      r.get("position"),
                "skills":     None,
                "avatar_url": None,
                "score":      0,
            } for r in (results[:top_k] if isinstance(results, list) else [])]
        except Exception as ex:
            raise HTTPException(status_code=500, detail=f"coworkers 検索のフォールバックに失敗しました: {ex}") from ex

    debug_sql = {"raw": raw_sql, "sanitized": safe_sql}
    return candidates, debug_sql

def _people_plan_queries(question: str, coworker_id: int | None) -> tuple[str, list[str]]:
    """
    LLMに「質問→最大3つの検索クエリ」と「短い前置き文」をJSONで作らせる。
    返り値: (narrative, queries)
    """
    if not chat:
        raise HTTPException(status_code=500, detail="OpenAI (chat) が初期化されていません")

    hints = []
    if coworker_id is not None and coworker_id > 0:
        hints.append(f"手元の名刺を持っている同僚IDが {coworker_id} なら、その同僚が関わっていそうな候補を優先して良い。")

    system = (
        "あなたは名刺DBから適切な人物を探すための検索プランナーです。"
        "ユーザーの自然文の質問を読み取り、名刺DBに投げる日本語の検索クエリを最大3個、簡潔に作ってください。"
        "各クエリは『会社名/部署/役割/地域/キーワード』などを含む短文で構いません。"
        "また、最初に短い前置き（どのジャンル・立場の人に当たるべきか）も作ってください。"
        "出力は必ず JSON 一個のみで、以下のスキーマに正確に従ってください："
        '{"narrative":"...","queries":["...","..."]}'
    )
    user = (
        f"ユーザーの質問: {question}\n"
        + (f"ヒント: {', '.join(hints)}\n" if hints else "")
        + "必ず JSON だけを返してください。余計な文章・コードフェンスは不要です。"
    )

    resp = chat.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    text = str(resp.content).strip()

    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        # JSONが取れない時は質問そのものを1クエリにする（前処理付き）
        return "以下の観点で該当しそうな人物を探索します。", [_cleanup_nl_query(question)]

    try:
        data = json.loads(m.group(0))
        narrative = (data.get("narrative") or "").strip() or "以下の観点で該当しそうな人物を探索します。"

        # ① JSONから取り出し
        queries = [q for q in (data.get("queries") or []) if isinstance(q, str) and q.strip()]
        if not queries:
            queries = [question]

        # ② クレンジング
        queries = [_cleanup_nl_query(q) for q in queries]
        queries = [q for q in queries if q] or [_cleanup_nl_query(question)]

        return narrative, queries[:3]
    except Exception:
        # 壊れたJSONでも安全に
        return "以下の観点で該当しそうな人物を探索します。", [_cleanup_nl_query(question)]

@app.post("/api/people/ask", response_model=PeopleAskResponse)
async def people_ask_endpoint(req: PeopleAskRequest):
    """
    例）{ "question": "富山県で行ったこのプロジェクトに詳しそうな人は？", "top_k": 5 }
    レスポンス：前置き文 + LLMが立てたクエリ配列 + 名刺DBの候補
    """
    if not (req.question or "").strip():
        return UTF8JSONResponse(PeopleAskResponse(narrative="質問が空です。", queries=[], candidates=[]).dict())

    # 1) LLMでクエリ計画
    narrative, queries = _people_plan_queries(req.question.strip(), req.coworker_id)

    # 2) 各クエリを人物検索にかけて集約
    all_candidates: dict[int, dict] = {}
    per_query_limit = max(1, req.top_k)  # 各クエリで十分拾う
    for q in queries:
        try:
            cand, _dbg = _people_search_core(q, per_query_limit, req.coworker_id)
        except HTTPException as e:
            # 1クエリ失敗しても他を続行
            print(f"[people/ask] query failed: {q} -> {e.detail}")
            continue
        for c in cand:
            cid = c.get("id")
            if cid is None:
                continue
            ex = all_candidates.get(cid)
            if ex is None or (c.get("score", 0) > (ex.get("score", 0) or 0)):
                all_candidates[cid] = c

    # 3) スコア順に並べ替えて上位を返す
    merged = sorted(all_candidates.values(), key=lambda x: (x.get("score") or 0, x.get("id") or 0), reverse=True)
    merged = merged[: req.top_k]

    return UTF8JSONResponse(PeopleAskResponse(narrative=narrative, queries=queries, candidates=merged).dict())

# ===== 会社情報 API =====
from pydantic import BaseModel

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

@app.get("/api/companies/by-name", response_model=CompanyInfoResponse)
async def api_company_by_name(name: str):
    """
    会社名で 1 件だけ取得。まずは完全一致、無ければ緩めの一致で拾う。
    参照DBは junk_db の companies。
    """
    try:
        sql1 = """
            SELECT id, name, corporate_number, location, company_type,
                   founding_date, capital, employee_number, business_summary
            FROM companies
            WHERE name = %s
            LIMIT 1
        """
        rows = _execute_query_db2(sql1, (name,))
        if not rows:
            # ()/（）/株式会社 の揺れを吸収したゆるめ検索
            sql2 = """
                SELECT id, name, corporate_number, location, company_type,
                       founding_date, capital, employee_number, business_summary
                FROM companies
                WHERE REPLACE(REPLACE(REPLACE(name,'株式会社',''),'（','('),'）',')')
                      LIKE CONCAT('%', REPLACE(REPLACE(REPLACE(%s,'株式会社',''),'（','('),'）',')'), '%')
                ORDER BY id ASC
                LIMIT 1
            """
            rows = _execute_query_db2(sql2, (name,))

        if not rows:
            return {"company": None}

        r = rows[0]
        return {
            "company": {
                "id": r.get("id"),
                "name": r.get("name"),
                "corporate_number": r.get("corporate_number"),
                "location": r.get("location"),
                "company_type": r.get("company_type"),
                "founding_date": r.get("founding_date"),
                "capital": r.get("capital"),
                "employee_number": r.get("employee_number"),
                "business_summary": r.get("business_summary"),
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- gBizINFO helpers: ここから追加（/detail の直前に置く） -----------------
def _fetch_gbiz_by_number(corporate_number: str):
    base = (GBIZINFO_URL or "").rstrip("/") or "https://info.gbiz.go.jp/hojin/v1/hojin"
    url = f"{base}/{corporate_number}"
    resp = requests.get(
        url,
        headers={"X-hojinInfo-api-token": GBIZINFO_API_KEY or "", "accept": "application/json"},
        timeout=10,
    )
    print("[gBizINFO:number]", resp.status_code, url)
    try:
        return resp.status_code, resp.json()
    except Exception:
        return resp.status_code, None

def _fetch_gbiz_by_name(company_name: str):
    # ざっくり正規化（ゆらぎ吸収）
    norm = str(company_name or "").replace("　", " ")
    for t in ("株式会社", "（株）", "(株)"):
        norm = norm.replace(t, "")
    norm = norm.strip()

    base = (GBIZINFO_URL or "").rstrip("/") or "https://info.gbiz.go.jp/hojin/v1/hojin"
    from urllib.parse import urlencode
    url = f"{base}?{urlencode({'name': norm})}"
    resp = requests.get(
        url,
        headers={"X-hojinInfo-api-token": GBIZINFO_API_KEY or "", "accept": "application/json"},
        timeout=10,
    )
    print("[gBizINFO:name]", resp.status_code, url)
    try:
        return resp.status_code, resp.json()
    except Exception:
        return resp.status_code, None

def _extract_gbiz_info(payload):
    if not payload:
        return None
    arr = payload.get("hojin-infos") or payload.get("hojinInfos") or []
    if not arr:
        return None
    info = arr[0]
    return {
        "corporate_number": info.get("corporate_number"),
        "name": info.get("name"),
        "location": info.get("location"),
        "postal_code": info.get("postal_code"),
        "company_type": info.get("qualification_grade"),
        "founding_date": info.get("date_of_establishment"),
        "capital": info.get("capital_stock"),
        "employee_number": info.get("employee_number"),
        "business_summary": info.get("business_summary"),
        "update_date": info.get("update_date"),
    }
# --- gBizINFO helpers: ここまで追加 -----------------------------------------

@app.get("/detail/{card_id}")
def get_detail(card_id: int):
    # --- 名刺情報（MySQL: junk_db） ---
    sql_card = """
        SELECT bc.id, bc.name, bc.company, bc.department, bc.position, bc.memo,
               bc.company_id, c.corporate_number
        FROM business_cards bc
        LEFT JOIN companies c ON bc.company_id = c.id
        WHERE bc.id = %s
    """
    rows = _execute_query_db2(sql_card, (card_id,))
    if not rows:
        raise HTTPException(404, "card not found")
    card = rows[0]

    # --- 同僚（同じ会社の他カード） ---
    sql_coworkers = """
        SELECT bc.id, bc.name, bc.department, bc.position
        FROM business_cards bc
        WHERE bc.company_id = %s AND bc.id <> %s
        ORDER BY bc.id
    """
    coworkers = _execute_query_db2(sql_coworkers, (card["company_id"], card_id))

    # --- gBizINFO: ①法人番号 → ②社名フォールバック ---
    gbiz_info = None
    gbiz_debug = {}
    try:
        corp = str(card.get("corporate_number") or "").strip()
        if not GBIZINFO_API_KEY:
            gbiz_debug["reason"] = "GBIZINFO_API_KEY not set"
        else:
            if corp:
                sc, payload = _fetch_gbiz_by_number(corp)
                gbiz_debug["number_status"] = sc
                gbiz_info = _extract_gbiz_info(payload)

            if not gbiz_info and (card.get("company") or "").strip():
                name = str(card["company"]).strip()
                sc2, payload2 = _fetch_gbiz_by_name(name)
                gbiz_debug["name_status"] = sc2
                gbiz_debug["queried_name"] = name
                gbiz_info = _extract_gbiz_info(payload2)
    except Exception as e:
        print(f"[gBizINFO] error: {e}")
        gbiz_debug["error"] = str(e)

    return {
        "person": card,
        "coworkers": coworkers,
        "gbiz_info": gbiz_info,
        "gbiz_debug": gbiz_debug,   # ← デバッグ用ヒント（UIでは非表示でもOK）
        "network": _get_network_for_card(card_id),
    }

@app.get("/gbizinfo/detail/{card_id}")
def get_gbizinfo_detail(card_id: int):
    """
    gBizINFO の取得状況だけを検証するための専用EP。
    見つからなければ 404 を返す（原因切り分け用）。
    """
    sql = """
        SELECT bc.company, c.corporate_number
        FROM business_cards bc
        LEFT JOIN companies c ON bc.company_id = c.id
        WHERE bc.id = %s
    """
    rows = _execute_query_db2(sql, (card_id,))
    if not rows:
        raise HTTPException(404, "card not found")

    corp = str(rows[0].get("corporate_number") or "").strip()
    name = str(rows[0].get("company") or "").strip()

    if not GBIZINFO_API_KEY:
        raise HTTPException(500, "GBIZINFO_API_KEY not set")

    # ① 法人番号
    if corp:
        sc, payload = _fetch_gbiz_by_number(corp)
        info = _extract_gbiz_info(payload)
        if info:
            return {"gbiz_info": info, "via": "number", "status": sc}

    # ② 社名
    if name:
        sc2, payload2 = _fetch_gbiz_by_name(name)
        info2 = _extract_gbiz_info(payload2)
        if info2:
            return {"gbiz_info": info2, "via": "name", "status": sc2, "queried_name": name}

    raise HTTPException(404, "gBizINFO not found for this corporate number")

@app.get("/api/coworkers/{coworker_id}/profile")
def get_coworker_profile(coworker_id: int):
    # --- 基本情報（title列→無ければposition列にフォールバック） ---
    sql_basic_title = """
        SELECT cw.id, cw.name, cw.title AS title, d.name AS department
        FROM coworkers cw
        LEFT JOIN departments d ON d.id = cw.department_id
        WHERE cw.id = %s
    """
    sql_basic_position = """
        SELECT cw.id, cw.name, cw.position AS title, d.name AS department
        FROM coworkers cw
        LEFT JOIN departments d ON d.id = cw.department_id
        WHERE cw.id = %s
    """

    try:
        rows = _execute_query_db2(sql_basic_title, (coworker_id,))
    except mysql.connector.errors.ProgrammingError as e:
        # 1054: Unknown column 'cw.title' の場合は position で再実行
        if "1054" in str(e):
            rows = _execute_query_db2(sql_basic_position, (coworker_id,))
        else:
            raise

    if not rows:
        raise HTTPException(status_code=404, detail="coworker not found")

    basic = rows[0]

    # --- 経歴 ---
    work_history = []
    for r in _execute_query_db2("""
        SELECT start_year, end_year, company, role, notes
        FROM coworker_experiences
        WHERE coworker_id = %s
        ORDER BY COALESCE(start_year, 0), COALESCE(end_year, 9999)
    """, (coworker_id,)):
        sy = r.get("start_year"); ey = r.get("end_year")
        period = f"{sy or ''}–{ey or ''}".strip("–")
        text = " / ".join([x for x in [r.get("company"), r.get("role"), r.get("notes")] if x])
        work_history.append({"period": period, "text": text})

    # --- プロジェクト履歴 ---
    project_history = []
    for r in _execute_query_db2("""
        SELECT year, title, description
        FROM coworker_projects
        WHERE coworker_id = %s
        ORDER BY COALESCE(year, 0)
    """, (coworker_id,)):
        period = str(r.get("year") or "")
        text = " / ".join([x for x in [r.get("title"), r.get("description")] if x])
        project_history.append({"period": period, "text": text})

    return {
        "id": basic["id"],
        "name": basic["name"],
        "title": basic.get("title"),
        "department": basic.get("department"),
        "work_history": work_history,
        "project_history": project_history,
    }