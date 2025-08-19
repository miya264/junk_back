from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import json
from pydantic import BaseModel
from typing import Optional, List
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.schema import SystemMessage, HumanMessage
from langchain_core.documents import Document
from pinecone import Pinecone
import uuid
import json
from datetime import datetime
from policy_agents import PolicyAgentSystem
from flexible_policy_agents import FlexiblePolicyAgentSystem


# 環境変数の読み込み
load_dotenv()

app = FastAPI(title="AI Agent API", version="1.0.0")

# UTF-8エンコーディングを明示的に設定
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
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

# モデルの初期化
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

# Pinecone初期化
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# 政策立案エージェントシステムの初期化
policy_system = PolicyAgentSystem(chat, embedding_model, index)
flexible_policy_system = FlexiblePolicyAgentSystem(chat, embedding_model, index)

# Pydanticモデル
class MessageRequest(BaseModel):
    content: str
    search_type: Optional[str] = "normal"  # "normal", "fact", "network"
    flow_step: Optional[str] = None  # "analysis", "objective", "concept", "plan", "proposal"
    context: Optional[dict] = None
    session_id: Optional[str] = None
    project_id: Optional[str] = None

class PolicyStepRequest(BaseModel):
    content: str
    step: str  # "analysis", "objective", "concept", "plan", "proposal"
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

# セッション管理（簡易版）
sessions = {}

def rerank_documents(query: str, docs: list[Document], chat: ChatOpenAI, top_k: int = 5) -> list[Document]:
    """文書の再ランキング"""
    prompt = PromptTemplate(
        input_variables=["query", "documents"],
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

def perform_rag_search(query: str) -> str:
    """RAG検索を実行"""
    print(f"DEBUG: perform_rag_search called with query: {query}")
    try:
        # ベクトル検索
        print("DEBUG: Starting vector search...")
        query_embedding = embedding_model.embed_query(query)
        print(f"DEBUG: Query embedding completed, length: {len(query_embedding)}")
        results = index.query(
            vector=query_embedding,
            top_k=15,
            include_metadata=True
        )
        print(f"DEBUG: Vector search completed, found {len(results['matches'])} matches")

        # Documentオブジェクトに変換（詳細な出典情報を保持）
        initial_docs = []
        for match in results['matches']:
            doc = Document(
                page_content=match['metadata']['text'],
                metadata={
                    'source': match['metadata']['source'],
                    'figure_section': match['metadata'].get('figure_section', ''),
                    'chunk_index': match['metadata'].get('chunk_index', ''),
                    'page_number': match['metadata'].get('page_number', ''),
                    'section_title': match['metadata'].get('section_title', ''),
                    'document_title': match['metadata'].get('document_title', ''),
                    'year': match['metadata'].get('year', ''),
                    'score': match['score']
                }
            )
            initial_docs.append(doc)

        # 再ランキング
        top_docs = rerank_documents(query, initial_docs, chat, top_k=5)

        # 出典付きテキストに変換（実際のソース名で表示）
        documents_string = ""
        source_references = []  # ソース参照用リスト
        
        for i, doc in enumerate(top_docs, 1):
            # 主要なソース識別情報を構築
            source = doc.metadata.get("source", f"文書_{i}")
            doc_title = doc.metadata.get("document_title", "")
            year = doc.metadata.get("year", "")
            section_title = doc.metadata.get("section_title", "")
            page_number = doc.metadata.get("page_number", "")
            
            # ソース名を構築（ユーザーが特定しやすい形式）
            if doc_title:
                primary_source = doc_title
            else:
                primary_source = source
            
            # 年度情報を追加
            if year:
                primary_source += f"({year})"
            
            # セクション情報を追加
            if section_title:
                primary_source += f" - {section_title}"
            
            # ページ情報を追加
            if page_number:
                primary_source += f" p.{page_number}"
            
            source_references.append(primary_source)
            
            # 詳細情報（参考用）
            detail_info = []
            detail_info.append(f"ファイル名: {source}")
            
            if doc.metadata.get("figure_section"):
                detail_info.append(f"図表: {doc.metadata.get('figure_section')}")
            
            score = doc.metadata.get("score", 0)
            detail_info.append(f"関連度: {score:.3f}")
            
            detail_text = " | ".join(detail_info)
            
            documents_string += f"【{primary_source}】\n詳細: {detail_text}\n{doc.page_content.strip()}\n\n"

        # プロンプトテンプレート
        combined_prompt = PromptTemplate(
            template="""あなたは思考整理をサポートする壁打ち相手です。ユーザーからの質問に対して、検索した情報を活用しながら自然な対話で応答してください。

【基本姿勢】
- 情報を一方的に伝えるのではなく、ユーザーの考えを引き出す
- 検索した情報を参考に、より深く考えるための質問を投げかける
- **必ず出典を明示し**、ユーザーが情報元を確認できるようにする
- 答えを出すのではなく、ユーザーが自分で気づけるよう導く

【出典明示の重要ルール】
- 情報を引用する際は**必ず実際のソース名を使用する**（例：「【令和5年度経済白書 - 第3章地域経済】によると〜」）
- 「出典1」「出典2」のような番号は使わず、ユーザーが実際に確認できるソース名を明記する
- 複数のソースから情報を組み合わせる場合も、それぞれの具体的なソース名を明示する
- ソース名にはタイトル、年度、章・節、ページ情報が含まれているので、ユーザーが元資料を特定できる

【情報の活用方法】
- 「【○○白書(2023) - △△章】では〜という事例がありますが、どう思われますか？」
- 「【××調査報告書 p.45】の情報を見ると〜とありますが、あなたの状況と比べていかがでしょう？」
- 「これらの資料から、何か気づくことはありますか？」

検索文書（詳細な出典情報付き）:
{document}

ユーザーからの質問: {query}

上記を踏まえ、**具体的なソース名を明示しながら**自然な対話で応答してください：""",
            input_variables=["document", "query"]
        )

        # Chat実行
        messages = [
            SystemMessage(content="あなたは思考整理をサポートする壁打ち相手です。情報を活用する際は必ず具体的なソース名（文書名、年度、章・節、ページ）を明示し、ユーザーが実際の情報元を確認できるようにしてください。答えではなく気づきを促すことを心がけてください。"),
            HumanMessage(content=combined_prompt.format(document=documents_string, query=query))
        ]

        response = chat.invoke(messages)
        return response.content.strip()
        
    except Exception as e:
        return f"RAG検索中にエラーが発生しました: {str(e)}"

def perform_policy_step(content: str, step: str, context: dict = None) -> str:
    """政策立案ステップ別処理"""
    try:
        result = policy_system.process_step(step, content, context)
        return result
    except Exception as e:
        return f"政策立案エージェント処理中にエラーが発生しました: {str(e)}"

def perform_normal_chat(content: str, session_id: str = None) -> str:
    """通常のチャット（自然な対話型壁打ち相手）"""
    try:
        # セッション情報がある場合、ファクト検索結果を取得
        fact_context = ""
        if session_id:
            session_state = flexible_policy_system._get_session_state(session_id)
            if session_state["fact_search_results"]:
                recent_facts = "\\n\\n".join(session_state["fact_search_results"][-2:])  # 最新2件
                fact_context = f"\\n\\n【これまでのファクト検索結果】\\n{recent_facts}"

        user_input_with_context = content + fact_context if fact_context else content
        
        messages = [
            SystemMessage(content="""あなたは思考整理をサポートする壁打ち相手です。ユーザーとの自然な対話を通じて、以下の役割を果たしてください：

【基本姿勢】
- 親しみやすく、自然な会話を心がける
- 答えや結論は出さず、ユーザーの考えを引き出す
- 質問や示唆を通じて、ユーザー自身が気づけるよう導く
- 構造化された回答ではなく、普通の会話として応答する

【対話の進め方】
1. まずはユーザーの話をしっかり聞く
2. 興味深い点や気になる点について質問する
3. 別の角度からの視点を提供する
4. 見落としがちなポイントを指摘する
5. より深く考えるための質問を投げかける

【ファクト検索結果の活用】
- これまでにファクト検索で得られた情報があれば、具体的なソース名を含めて「そういえば、先ほど【○○白書(2023)】で〜という情報がありましたが」という形で自然に言及
- ファクト情報をもとに「【××調査報告書 - △△章】の情報を見て、どう思われますか？」といった質問を投げかける
- 具体的なソース名を示すことで、ユーザーが実際の資料を確認できるようサポートする

【最終的な目標】
対話を重ねる中で、ユーザーが自然に以下の3つの観点を整理できるようになること：
- 課題と裏付け（定量・定性データ）
- 課題の背景にある構造（制度・市場環境など）
- 解決すべき課題の優先度と理由

ただし、これらを無理に構造化しようとせず、自然な対話の流れを大切にしてください。"""),
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
        # ユーザーメッセージの作成
        user_message = MessageResponse(
            id=str(uuid.uuid4()),
            content=request.content,
            type="user",
            timestamp=datetime.now().isoformat(),
            search_type=request.search_type
        )
        
        # AI応答の生成
        print(f"DEBUG: search_type = {request.search_type}")
        print(f"DEBUG: content = {request.content}")
        if request.search_type == "fact":
            print("DEBUG: Executing RAG search")
            ai_content = perform_rag_search(request.content)
            print(f"DEBUG: RAG result = {ai_content}")
            
            # ファクト検索結果をセッションに保存
            if request.session_id:
                flexible_policy_system.add_fact_search_result(request.session_id, ai_content)
                print(f"DEBUG: Added fact search result to session {request.session_id}")
        elif request.search_type == "network":
            # 人脈検索の処理（将来の拡張用）
            ai_content = perform_normal_chat(request.content)
        elif request.flow_step:
            # 柔軟な政策立案ステップ別処理
            session_id = request.session_id or str(uuid.uuid4())
            project_id = request.project_id
            
            result = flexible_policy_system.process_step_flexible(
                request.flow_step, 
                request.content, 
                session_id, 
                project_id
            )
            
            if "error" in result:
                ai_content = result["error"]
            else:
                ai_content = result["result"]
        else:
            ai_content = perform_normal_chat(request.content, request.session_id)
        
        ai_message = MessageResponse(
            id=str(uuid.uuid4()),
            content=ai_content,
            type="ai",
            timestamp=datetime.now().isoformat(),
            search_type=request.search_type
        )
        
        return UTF8JSONResponse(ai_message.dict())
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/policy-step", response_model=PolicyStepResponse)
async def policy_step_endpoint(request: PolicyStepRequest):
    """政策立案ステップ別処理エンドポイント"""
    try:
        # ステップ別処理の実行
        ai_content = perform_policy_step(request.content, request.step, request.context)
        
        response = PolicyStepResponse(
            id=str(uuid.uuid4()),
            content=ai_content,
            step=request.step,
            timestamp=datetime.now().isoformat(),
            context=request.context
        )
        
        return UTF8JSONResponse(response.dict())
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/policy-flexible", response_model=FlexiblePolicyResponse)
async def flexible_policy_endpoint(request: MessageRequest):
    """柔軟な政策立案エンドポイント"""
    try:
        if not request.flow_step:
            raise HTTPException(status_code=400, detail="flow_step is required")
        
        session_id = request.session_id or str(uuid.uuid4())
        project_id = request.project_id
        
        result = flexible_policy_system.process_step_flexible(
            request.flow_step,
            request.content,
            session_id,
            project_id
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        response = FlexiblePolicyResponse(
            id=str(uuid.uuid4()),
            content=result["result"],
            step=result["step"],
            timestamp=datetime.now().isoformat(),
            session_id=result["session_id"],
            project_id=result.get("project_id"),
            full_state=result["full_state"]
        )
        
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

@app.get("/")
async def root():
    """ルートエンドポイント"""
    return {"message": "AI Agent API is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 