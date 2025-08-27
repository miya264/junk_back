from fastapi import APIRouter, HTTPException, Response, Cookie
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import asyncio
import json
import uuid

from core.config import chat, embedding_model, index, flexible_policy_system
from core.cache import memory_cache
from DB.mysql.mysql_crud import (
    save_project_step_sections,
    get_project_step_sections,
    create_project,
    get_project_by_id,
    search_all_projects,
    get_projects_by_coworker,
    search_coworkers,
    CRUDError,
)
from langchain.prompts import PromptTemplate
from langchain.schema import SystemMessage, HumanMessage
from langchain_core.documents import Document
from datetime import datetime, timezone, timedelta

router = APIRouter(prefix="/api/chat", tags=["chat"])

JST = timezone(timedelta(hours=9))

def get_jst_now():
    return datetime.now(JST)

class UTF8JSONResponse(JSONResponse):
    def __init__(self, content, **kwargs):
        kwargs.setdefault('media_type', 'application/json; charset=utf-8')
        super().__init__(content, **kwargs)

class MessageRequest(BaseModel):
    content: str
    search_type: Optional[str] = "normal"
    flow_step: Optional[str] = None
    context: Optional[dict] = None
    session_id: Optional[str] = None
    project_id: Optional[str] = None

class MessageResponse(BaseModel):
    id: str
    content: str
    type: str
    timestamp: str
    search_type: Optional[str] = None

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

@memory_cache(ttl_seconds=600)
async def perform_rag_search(query: str) -> tuple[str, list[dict]]:
    try:
        if not embedding_model or not index:
            return "申し訳ございませんが、現在RAG検索機能は利用できません。環境設定を確認してください。", []

        dense_embedding = await embedding_model.aembed_query(query)
        
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

async def perform_normal_chat(content: str, session_id: str = None) -> str:
    try:
        if not chat:
            return "申し訳ございませんが、現在チャット機能は利用できません。環境設定を確認してください。"
        
        fact_context = ""
        if session_id and flexible_policy_system:
            try:
                session_state = await flexible_policy_system.get_session_state(session_id)
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

@router.post("/", response_model=MessageResponse)
async def chat_endpoint(request: MessageRequest):
    try:
        sid = request.session_id or str(uuid.uuid4())
        session_id, step_id = await _ensure_chat_session(sid, request.project_id, request.flow_step)
        await _save_chat_message(session_id, request.project_id, step_id, 'user', request.search_type or 'normal', request.content)

        if request.search_type == "fact":
            if not (embedding_model and index):
                ai_content = "申し訳ございませんが、現在RAG検索機能は利用できません。環境設定を確認してください。"
                sources = []
            else:
                ai_content, sources = await perform_rag_search(request.content)
            
            if request.session_id and flexible_policy_system:
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
            # ... (人脈検索ロジックはrouters/people.pyに移動)
            # 簡略化
            ai_content = await perform_normal_chat(request.content, request.session_id)

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