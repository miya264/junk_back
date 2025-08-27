from fastapi import FastAPI, HTTPException, Response, Cookie, Depends
from fastapi.middleware.cors import CORSMiddleware
import asyncio
from fastapi.responses import JSONResponse
try:
    from rag.company_vector import search_companies
except Exception as e:
    print(f"Warning: Could not import company_vector: {e}")
    def search_companies(query, top_k=8):
        return []

class UTF8JSONResponse(JSONResponse):
    media_type = "application/json; charset=utf-8"
import json
from pydantic import BaseModel, Field
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
import urllib.parse
from routers.routers_people import router as people_router

try:
    from pinecone import Pinecone
except Exception as e:
    print(f"Pinecone import error: {e}")
    Pinecone = None
import uuid
from datetime import datetime, timezone, timedelta
try:
    from policy_agents import PolicyAgentSystem
except ImportError:
    PolicyAgentSystem = None

try:
    from flexible_policy_agents import FlexiblePolicyAgentSystem, CrudSectionRepo, CrudChatRepo
except ImportError:
    FlexiblePolicyAgentSystem = None
    CrudSectionRepo = None
    CrudChatRepo = None
    
import asyncmy
from asyncmy.errors import Error
import requests
import re

ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=False)

app = FastAPI(title="AI Agent API", version="1.0.0")

CACHE = {}
CACHE_TTL = 300

def cache_key(*args, **kwargs):
    key_string = str(args) + str(sorted(kwargs.items()))
    return hashlib.md5(key_string.encode()).hexdigest()

def memory_cache(ttl_seconds=CACHE_TTL):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            key = f"{func.__name__}_{cache_key(*args, **kwargs)}"
            current_time = time.time()
            
            if key in CACHE:
                cached_data, timestamp = CACHE[key]
                if current_time - timestamp < ttl_seconds:
                    return cached_data
                else:
                    del CACHE[key]
            
            result = await func(*args, **kwargs)
            CACHE[key] = (result, current_time)
            return result
        return wrapper
    return decorator

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

JST = timezone(timedelta(hours=9))

def get_jst_now():
    return datetime.now(JST)

origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://apps-junk-02.azurewebsites.net",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", "http://localhost:3001", "http://localhost:3002", "http://localhost:3003", "http://localhost:3004",
        "http://127.0.0.1:3000", "http://127.0.0.1:3001", "http://127.0.0.1:3002", "http://127.0.0.1:3003", "http://127.0.0.1:3004",
        "https://apps-junk-02.azurewebsites.net",
        "https://aps-junk-01-fbgncnexhuekadft.canadacentral-01.azurewebsites.net",
        "https://apps-junk-01.azurewebsites.net",
    ],
    allow_origin_regex=r"https://.*\.azurewebsites\.net",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "rag-hakusho")
GBIZINFO_API_KEY = os.getenv("GBIZINFO_API_KEY")
GBIZINFO_URL = os.getenv("GBIZINFO_URL", "https://info.gbiz.go.jp/hojin/v1/hojin")

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

print(f"PINECONE_API_KEY present: {'YES' if PINECONE_API_KEY else 'NO'}")
print(f"Pinecone module loaded: {'YES' if Pinecone else 'NO'}")

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

policy_system = None
flexible_policy_system = None
from DB.mysql.mysql_crud import MySQLCRUD

if chat and embedding_model and index and FlexiblePolicyAgentSystem:
    try:
        main_crud = MySQLCRUD()
        if CrudSectionRepo and CrudChatRepo:
            section_repo = CrudSectionRepo(main_crud)
            chat_repo = CrudChatRepo(main_crud)
        else:
            section_repo = None
            chat_repo = None
        
        if FlexiblePolicyAgentSystem and section_repo and chat_repo:
            flexible_policy_system = FlexiblePolicyAgentSystem(chat, section_repo, chat_repo)
        print("✓ Policy agent systems initialized successfully")
    except Exception as e:
        print(f"⚠️ Warning: Policy agent initialization failed: {e}")
else:
    print("⚠️ Warning: Policy agents not available - AI services not initialized")

@app.on_event("startup")
async def startup_event():
    try:
        from DB.mysql.mysql_connection import init_db_pool
        await init_db_pool()
        print("✓ Database connection pool initialized successfully")
    except Exception as e:
        print(f"⚠️ Warning: Database connection failed: {e}")
        print("⚠️ Server will start without database connectivity")

async def _execute_query(query: str, params: tuple | None = None):
    from DB.mysql.mysql_connection import get_mysql_db
    db = get_mysql_db()
    result = await db.execute_query(query, params)
    return result

async def _get_step_id(project_id: Optional[str], step_key: Optional[str]) -> Optional[str]:
    if not project_id or not step_key:
        return None
    rows = await _execute_query(
        "SELECT id FROM policy_steps WHERE project_id = %s AND step_key = %s",
        (project_id, step_key),
    )
    if rows:
        return rows[0]['id']
    new_id = str(uuid.uuid4())
    await _execute_query(
        """
        INSERT INTO policy_steps (id, project_id, step_key, step_name, order_no, status)
        VALUES (%s, %s, %s, %s, 1, 'active')
        """,
        (new_id, project_id, step_key, step_key.title()),
    )
    return new_id

async def _ensure_chat_session(session_id: str, project_id: Optional[str], step_key: Optional[str]) -> tuple[str, Optional[str]]:
    if not project_id:
        return session_id, None
    step_id = await _get_step_id(project_id, step_key) if (project_id and step_key) else None
    await _execute_query(
        """
        INSERT IGNORE INTO chat_sessions (id, project_id, step_id, title, created_by, created_at, updated_at)
        VALUES (%s, %s, %s, NULL, NULL, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        """,
        (session_id, project_id, step_id),
    )
    return session_id, step_id

async def _save_chat_message(session_id: str, project_id: Optional[str], step_id: Optional[str], role: str, msg_type: str, content: str) -> None:
    if not project_id:
        return
    await _execute_query(
        """
        INSERT INTO chat_messages (id, session_id, project_id, step_id, role, msg_type, content, created_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
        """,
        (str(uuid.uuid4()), session_id, project_id, step_id, role, msg_type, content),
    )

async def rerank_documents(query: str, docs: list[Document], chat: ChatOpenAI, top_k: int = 5) -> list[Document]:
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
    response = await chat.ainvoke([msg])
    selected = []
    for s in response.content.split(","):
        try:
            idx = int(s.strip())
            if 0 <= idx < len(docs):
                selected.append(docs[idx])
        except:
            continue
    return selected

@memory_cache(ttl_seconds=600)
async def perform_rag_search(query: str) -> tuple[str, list[dict]]:
    try:
        if not embedding_model or not index:
            return "申し訳ございませんが、現在RAG検索機能は利用できません。環境設定を確認してください。", []

        dense_embedding = await embedding_model.aembed_query(query)
        
        # ハイブリッド検索を実行（再ランキングを省略）
        results = await asyncio.to_thread(
            index.query,
            vector=dense_embedding,
            top_k=5, 
            include_metadata=True
        )

        matches = results.matches if hasattr(results, "matches") else results.get("matches", [])
        top_docs = []
        for m in matches:
            meta = m.metadata if hasattr(m, "metadata") else m.get("metadata", {}) or {}
            score = m.score if hasattr(m, "score") else m.get("score")
            text = meta.get("text") or meta.get("chunk") or meta.get("page_content") or ""
            if not text:
                continue
            doc = Document(
                page_content=text,
                metadata={
                    "source": meta.get("source", ""),
                    "document_title": meta.get("document_title", "不明"),
                    "year": meta.get("year", ""),
                    "section_title": meta.get("section_title", ""),
                    "score": score,
                },
            )
            top_docs.append(doc)

        documents_string = ""
        for i, doc in enumerate(top_docs, 1):
            source_info = f"【{doc.metadata.get('document_title', '不明')}"
            if doc.metadata.get('year'):
                source_info += f"（{doc.metadata['year']}年度）"
            if doc.metadata.get('section_title'):
                source_info += f" - {doc.metadata['section_title']}"
            source_info += "】"
            documents_string += f"{source_info}\n{doc.page_content.strip()}\n\n"

        prompt = PromptTemplate(
            template="""
あなたは思考整理をサポートする壁打ち相手です。ユーザーからの質問と、それに関連する複数の参照資料が提供されます。
これらの資料を活用して、ユーザーへの応答を作成してください。

【基本ルール】
1. 出典に基づく情報は、自然な文章で要約し、必ず本文中に【資料名 – 章節】の形式で出典を明記する。
2. 出典に存在しない情報は「この資料からは確認できません」と明言する。
3. 情報が一部しか見つからない場合は、「情報不足」として整理する。
4. 情報不足がある場合は、次のような具体的な探し方を提示する：
   - **専門家に聞く**：商工会議所の経営相談窓口、中小企業診断士協会、中小機構の「よろず支援拠点」に常駐するDXアドバイザー
   - **論文を読む**：CiNiiで「中小企業 DX 成功事例」、Google Scholarで「SME digital transformation Japan」などで検索
   - **統計を見る**：総務省「通信利用動向調査」、経産省「中小企業白書」、IPA「DX白書」
   - **実務者に聞く**：LinkedInで「DX推進担当」と肩書のある人、業界団体の交流会、自治体のDXセミナー
5. 最後に、ユーザーの考えを引き出すような問いかけで締めくくる。

【出力フォーマット】
●情報整理（参照資料に基づく要約）
●情報不足（確認できなかった内容）
●代替的な探し方（具体的に）
●投げかけ（ユーザーに次の一歩を考えさせる質問）

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

        response = await chat.ainvoke(messages)
        ai_content = response.content.strip()

        sources: list[dict] = []
        for doc in top_docs:
            sources.append({
                "source": doc.metadata.get("source"),
                "document_title": doc.metadata.get("document_title"),
                "year": doc.metadata.get("year"),
                "section_title": doc.metadata.get("section_title"),
                "score": doc.metadata.get("score"),
            })

        return ai_content, sources
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"RAG検索中にエラーが発生しました: {str(e)}", []

async def perform_policy_step(content: str, step: str, context: dict = None) -> str:
    try:
        if not policy_system:
            return "申し訳ございませんが、現在政策立案機能は利用できません。環境設定を確認してください。"
        
        result = await asyncio.to_thread(policy_system.process_step, step, content, context)
        return result
    except Exception as e:
        return f"政策立案エージェント処理中にエラーが発生しました: {str(e)}"

async def perform_normal_chat(content: str, session_id: str = None) -> str:
    try:
        if not chat:
            return "申し訳ございませんが、現在チャット機能は利用できません。環境設定を確認してください。"
        
        fact_context = ""
        if session_id and flexible_policy_system:
            try:
                session_state = flexible_policy_system.get_session_state(session_id)
                if session_state and session_state.get("fact_search_results"):
                    recent_facts = "\n\n".join(session_state["fact_search_results"][-2:])
                    fact_context = f"\n\n【これまでのファクト検索結果】\n{recent_facts}"
            except Exception as e:
                print(f"Warning: Failed to get session state: {e}")

        user_input_with_context = content + fact_context if fact_context else content
        
        messages = [
            SystemMessage(content="""あなたは思考整理をサポートする壁打ち相手です。ユーザーとの自然な対話を通じて、以下の役割を果たしてください：
- ユーザーの発言の意図を汲み取り、共感的な返答をする
- 思考を構造化するための質問を投げかけ、論点を整理する
- 曖昧な表現を具体的な言葉に置き換えるよう促す
- 結論を急がず、多角的な視点を提供し、ユーザーの気づきを支援する
- 最後に、次のアクションや思考を促すような問いかけで締めくくる"""),
            HumanMessage(content=user_input_with_context)
        ]
        
        response = await chat.ainvoke(messages)
        return response.content.strip()
        
    except Exception as e:
        return f"チャット処理中にエラーが発生しました: {str(e)}"

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
    navigate_to: Optional[str] = None
    type: Optional[str] = None
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
    sections: List[Dict[str, str]]

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

from DB.mysql.mysql_crud import (
    search_coworkers,
    create_project,
    get_project_by_id,
    search_all_projects,
    get_projects_by_coworker,
    save_project_step_sections,
    get_project_step_sections,
    health_check as db_health_check,
    CRUDError
)

from auth import auth_service, LoginRequest, LoginResponse
from auth import auth_service as auth_service_module

async def get_current_user_with_db(access_token: str = Cookie(None)):
    user = await auth_service_module.get_current_user(access_token)
    if not user:
        raise HTTPException(status_code=401, detail="認証が必要です")
    return user

sessions = {}

@app.post("/api/chat", response_model=MessageResponse)
async def chat_endpoint(request: MessageRequest):
    try:
        print(f"[DEBUG] Received request: content='{request.content}', search_type='{request.search_type}'")
        sid = request.session_id or str(uuid.uuid4())
        session_id, step_id = await _ensure_chat_session(sid, request.project_id, request.flow_step)
        await _save_chat_message(session_id, request.project_id, step_id, 'user', request.search_type or 'normal', request.content)

        print(f"[DEBUG] Checking conditions:")
        print(f"[DEBUG] request.search_type: '{request.search_type}' (type: {type(request.search_type)})")
        print(f"[DEBUG] request.flow_step: '{request.flow_step}' (type: {type(request.flow_step)})")
        print(f"[DEBUG] search_type == 'fact': {request.search_type == 'fact'}")
        print(f"[DEBUG] search_type == 'network': {request.search_type == 'network'}")
        print(f"[DEBUG] flow_step is truthy: {bool(request.flow_step)}")

        if request.search_type == "fact":
            if not (embedding_model and index):
                ai_content = "申し訳ございませんが、現在RAG検索機能は利用できません。環境設定を確認してください。"
                sources = []
            else:
                ai_content, sources = await perform_rag_search(request.content)
            
            if request.session_id and flexible_policy_system and hasattr(flexible_policy_system, "add_fact_search_result"):
                try:
                    flexible_policy_system.add_fact_search_result(request.session_id, ai_content)
                except Exception as e:
                    print(f"Warning: Failed to add fact search result: {e}")

            try:
                session_id, step_id = await _ensure_chat_session(sid, request.project_id, request.flow_step)
                if request.project_id and step_id:
                    await _execute_query(
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

        elif request.search_type == "network":
            print(f"[DEBUG] Network search triggered for: '{request.content}'")
            try:
                if not embedding_model or not index:
                    ai_content = {
                        "type": "company_search_result",
                        "query": request.content,
                        "companies": [],
                        "narrative": "現在ベクトル検索の初期化に失敗しています。環境変数をご確認ください。"
                    }
                else:
                    # 会社ベクトル検索（上位8件程度）
                    hits = search_companies(request.content.strip(), top_k=8)
                    companies = [{
                        "id": h.get("id"),  # companies.id（ある場合）
                        "name": h.get("name"),
                        "corporate_number": h.get("corporate_number"),
                        "location": h.get("location"),
                        "score": h.get("score", 0.0)
                    } for h in hits]

                    ai_content = {
                        "type": "company_search_result",
                        "query": request.content,
                        "companies": companies,
                        "narrative": f"「{request.content}」に関連する会社候補を{len(companies)}件見つけました。"
                    }
            except Exception as e:
                print(f"[company-search] error: {e}")
            ai_content = {
                "type": "company_search_result",
                "query": request.content,
                "companies": [],
                "narrative": "会社検索中にエラーが発生しました。キーワードを変えてお試しください。"
            }

        elif request.flow_step:
            if not flexible_policy_system:
                ai_content = "申し訳ございませんが、現在政策立案機能は利用できません。環境設定を確認してください。"
            else:
                session_id = request.session_id or str(uuid.uuid4())
                project_id = request.project_id
                
                try:
                    result = await flexible_policy_system.process_flexible(
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
            
        else:
            print(f"[DEBUG] Normal chat triggered for: '{request.content}', search_type='{request.search_type}'")
            if not chat:
                ai_content = "申し訳ございませんが、現在チャット機能は利用できません。環境設定を確認してください。"
            else:
                ai_content = await perform_normal_chat(request.content, request.session_id)
        
        if isinstance(ai_content, dict):
            ai_message = MessageResponse(
                id=str(uuid.uuid4()),
                content=json.dumps(ai_content, ensure_ascii=False),
                type="ai",
                timestamp=get_jst_now().isoformat(),
                search_type=request.search_type
            )
        else:
            ai_message = MessageResponse(
                id=str(uuid.uuid4()),
                content=ai_content,
                type="ai",
                timestamp=get_jst_now().isoformat(),
                search_type=request.search_type
            )
        db_content = json.dumps(ai_content, ensure_ascii=False) if isinstance(ai_content, dict) else ai_content
        await _save_chat_message(session_id, request.project_id, step_id, 'ai', request.search_type or 'normal', db_content)
        
        return UTF8JSONResponse(ai_message.dict())
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/policy-flexible", response_model=FlexiblePolicyResponse)
async def flexible_policy_endpoint(request: MessageRequest):
    try:
        if not request.flow_step:
            raise HTTPException(status_code=400, detail="flow_step is required")
        
        if not flexible_policy_system:
            raise HTTPException(status_code=503, detail="Policy system not available. Please check environment configuration.")
        
        session_id = request.session_id or str(uuid.uuid4())
        project_id = request.project_id
        
        try:
            result = await flexible_policy_system.process_flexible(
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
            timestamp=get_jst_now().isoformat(),
            session_id=session_id,
            project_id=project_id,
            navigate_to=result.get("navigate_to"),
            type=result.get("type"),
            full_state=result["full_state"]
        )

        session_id, step_id = await _ensure_chat_session(session_id, project_id, request.flow_step)
        await _save_chat_message(session_id, project_id, step_id, 'user', 'normal', request.content)
        await _save_chat_message(session_id, project_id, step_id, 'ai', 'normal', result["result"]) 
        
        return UTF8JSONResponse(response.dict())
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/session-state/{session_id}", response_model=SessionStateResponse)
async def get_session_state(session_id: str):
    try:
        state = await asyncio.to_thread(flexible_policy_system.get_session_state, session_id)
        
        if "error" in state:
            raise HTTPException(status_code=404, detail=state["error"])
        
        return SessionStateResponse(**state)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/sessions", response_model=List[ChatSession])
async def get_sessions():
    return list(sessions.values())

@app.post("/api/sessions", response_model=ChatSession)
async def create_session():
    session_id = str(uuid.uuid4())
    now = get_jst_now()
    
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
    try:
        print(f"DEBUG: Received request: project_id={request.project_id}, step_key={request.step_key}, sections_count={len(request.sections)}")
        print(f"DEBUG: Request sections: {request.sections}")
        
        saved_sections = await save_project_step_sections(request.project_id, request.step_key, request.sections)
        
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
    try:
        sections = await get_project_step_sections(project_id, step_key)
        return UTF8JSONResponse([ProjectStepSectionResponse(**section).dict() for section in sections])
    except CRUDError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/project-all-sections/{project_id}")
async def get_project_all_sections_endpoint(project_id: str):
    try:
        from DB.mysql.mysql_crud import MySQLCRUD
        crud = MySQLCRUD()
        sections_by_step = await crud.get_project_all_step_sections(project_id)
        return UTF8JSONResponse(sections_by_step)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/coworkers/search", response_model=List[CoworkerResponse])
async def search_coworkers_endpoint(q: str = "", department: str = ""):
    try:
        coworkers = await search_coworkers(q, department)
        return UTF8JSONResponse([CoworkerResponse(**coworker).dict() for coworker in coworkers])
    except CRUDError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/projects", response_model=ProjectResponse)
async def create_project_endpoint(request: ProjectCreateRequest):
    try:
        project = await create_project(
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
    try:
        project = await get_project_by_id(project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        return UTF8JSONResponse(ProjectResponse(**project).dict())
    except CRUDError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/projects", response_model=List[ProjectResponse])
async def search_projects_endpoint(q: str = "", limit: int = 50):
    try:
        projects = await search_all_projects(q, limit)
        return UTF8JSONResponse([ProjectResponse(**project).dict() for project in projects])
    except CRUDError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/projects/by-coworker/{coworker_id}", response_model=List[ProjectResponse])
async def get_projects_by_coworker_endpoint(coworker_id: int):
    try:
        projects = await get_projects_by_coworker(coworker_id)
        return UTF8JSONResponse([ProjectResponse(**project).dict() for project in projects])
    except CRUDError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/auth/login", response_model=LoginResponse)
async def login_endpoint(request: LoginRequest, response: Response):
    try:
        return await auth_service.login(request, response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/auth/logout")
async def logout_endpoint(response: Response):
    try:
        await auth_service.logout(response)
        return {"message": "Successfully logged out"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/auth/me")
async def get_current_user_endpoint(access_token: str = Cookie(None)):
    try:
        user = await auth_service.get_current_user(access_token)
        if not user:
            raise HTTPException(status_code=401, detail="認証が必要です")
        return user
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/auth/verify")
async def verify_token_endpoint(access_token: str = Cookie(None)):
    try:
        user = await auth_service.get_current_user(access_token)
        if not user:
            return {"valid": False}
        return {"valid": True, "user": user}
    except Exception as e:
        return {"valid": False, "error": str(e)}

@app.get("/")
async def root():
    return {"message": "AI Agent API is running"}

class PeopleSearchRequest(BaseModel):
    query: str = Field(..., description="自然文の検索条件（例：'富山 EC デザイン'）")
    top_k: int = Field(5, ge=1, le=50, description="最大件数")
    coworker_id: int | None = Field(None, description="優先したい同僚のID（任意）")

_ALLOWED_TABLE_NAMES = {"business_cards", "companies", "coworker_relations"}
def _people_sanitize_sql(sql: str, top_k: int) -> str:
    # ... (この関数は変更なし)
    s = sql.strip().rstrip(";")
    s = re.sub(r"\bbc\.title\b", "bc.position", s, flags=re.I)
    s = re.sub(r"\bbc\.avatar_url\b", "NULL", s, flags=re.I)
    low_raw = s.lower()
    banned_regexes = [
        r";", r"\binsert\b", r"\bupdate\b", r"\bdelete\b",
        r"\bdrop\b", r"\balter\b", r"\bcreate\b", r"\bgrant\b", r"\brevoke\b", r"\btruncate\b",
        r"\bunion\b", r"\bwith\b",
        r"\bin\s*\(\s*select\b", r"\bexists\s*\(", r"\(\s*select\b",
    ]
    for pat in banned_regexes:
        if re.search(pat, low_raw):
            raise HTTPException(status_code=400, detail="Unsafe SQL is not allowed")
    low = re.sub(r"\s+", " ", low_raw)
    num_selects = len(re.findall(r"\bselect\b", low_raw, flags=re.I))
    if num_selects != 1:
        raise HTTPException(status_code=400, detail="Only a single SELECT is allowed")
    if " from business_cards" not in low:
        raise HTTPException(status_code=400, detail="Root table must be business_cards")
    tables = re.findall(r"\b(from|join)\s+([a-zA-Z0-9_]+)", low)
    used_tables = {t[1] for t in tables}
    if not used_tables.issubset(_ALLOWED_TABLE_NAMES):
        bad = sorted(list(used_tables - _ALLOWED_TABLE_NAMES))
        raise HTTPException(status_code=400, detail=f"Disallowed tables detected: {bad}")
    if re.search(r"\blimit\b\s+\d+", s, flags=re.I):
        s = re.sub(r"\blimit\b\s+\d+", f"LIMIT {int(top_k)}", s, flags=re.I)
    else:
        s = s + f" LIMIT {int(top_k)}"
    return s

async def _people_generate_sql_with_llm(nl_query: str, top_k: int, coworker_id: int | None, force_single_select: bool, force_template: bool) -> str:
    # ... (この関数は非同期でそのまま利用)
    if not chat:
        raise HTTPException(status_code=500, detail="OpenAI (chat) が初期化されていません")
    
    schema_desc = []
    for tbl, cols in [
        ("business_cards", ["id", "name", "company", "department", "position", "memo", "owner_coworker_id", "corporate_number", "company_id"]),
        ("companies", ["id", "name", "corporate_number", "postal_code", "location", "company_type", "founding_date", "capital", "employee_number", "business_summary", "update_date"]),
        ("coworker_relations", ["coworker_id", "business_card_id", "first_contact_date", "last_contact_date", "contact_count"]),
    ]:
        schema_desc.append(f"- {tbl}({', '.join(cols)})")
    schema_text = "\n".join(schema_desc)
    
    rules = [
        "Generate EXACTLY ONE SELECT statement only. The word SELECT must appear EXACTLY ONCE.",
        "NO subqueries of any kind: NO EXISTS(...), NO IN (SELECT ...), NO (SELECT ...) in any expression, NO CTEs, NO UNION.",
        "The main table MUST be business_cards (alias bc).",
        "Use LEFT JOIN companies c ON c.id = bc.company_id when needed.",
        "Use LEFT JOIN coworker_relations cr ON cr.business_card_id = bc.id when needed.",
        "Use only allowed columns. Return these aliases: id, name, company, department, title, avatar_url, score",
        "company := COALESCE(c.name, bc.company), title := COALESCE(bc.title, bc.position)",
        "ALWAYS ORDER BY score DESC, id ASC.",
        "ALWAYS include LIMIT (the number will be overwritten later).",
        'Output MUST be a single JSON object: {"sql": "..."} . No extra text, no code fences.',
        "All text filters MUST use LIKE with wildcards: LIKE CONCAT('%', <keyword>, '%'). Never use equality (=) for company names or titles.",
        "When matching company names, normalize by stripping '株式会社', '(株)', and spaces using "
        "REPLACE(REPLACE(REPLACE(COALESCE(c.name, bc.company),'株式会社',''),'(株)',''),' ',''). "
        "Compare that normalized value with LIKE CONCAT('%', <company_keyword>, '%').",
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

    resp = await chat.ainvoke([SystemMessage(content=system), HumanMessage(content=user)])
    content_text = str(resp.content).strip()

    m = re.search(r"\{[\s\S]*\}", content_text)
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

    if last_err:
        raise last_err
    raise HTTPException(status_code=400, detail="SQL generation failed")


@app.post("/api/people/search")
async def people_search_endpoint(req: PeopleSearchRequest):
    if not (req.query or "").strip():
        return UTF8JSONResponse({"candidates": []})
    eff_coworker_id = req.coworker_id if (req.coworker_id and req.coworker_id > 0) else None
    safe_sql, raw_sql = await _generate_and_sanitize_people_sql(
        req.query.strip(), req.top_k, eff_coworker_id
    )
    from DB.mysql.mysql_connection import get_mysql_db
    try:
        db = get_mysql_db()
        if not db.connection_ready:
            print("Database not available, returning empty candidates list")
            candidates = []
        else:
            rows = await db.execute_query(safe_sql)
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
    except Exception as e:
        print(f"People search database error: {e}")
        candidates = []
    if not candidates:
        try:
            results = await search_coworkers(q=req.query, department="")
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
    return UTF8JSONResponse(res)

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)