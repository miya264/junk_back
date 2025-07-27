from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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
from datetime import datetime

# 環境変数の読み込み
load_dotenv()

app = FastAPI(title="AI Agent API", version="1.0.0")

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_methods=["*"],
    allow_headers=["*"],
)

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

# Pydanticモデル
class MessageRequest(BaseModel):
    content: str
    search_type: Optional[str] = "normal"  # "normal", "fact", "network"

class MessageResponse(BaseModel):
    id: str
    content: str
    type: str
    timestamp: datetime
    search_type: Optional[str] = None

class ChatSession(BaseModel):
    id: str
    title: str
    created_at: datetime
    updated_at: datetime

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
    try:
        # ベクトル検索
        query_embedding = embedding_model.embed_query(query)
        results = index.query(
            vector=query_embedding,
            top_k=15,
            include_metadata=True
        )

        # Documentオブジェクトに変換
        initial_docs = []
        for match in results['matches']:
            doc = Document(
                page_content=match['metadata']['text'],
                metadata={
                    'source': match['metadata']['source'],
                    'figure_section': match['metadata'].get('figure_section', ''),
                    'chunk_index': match['metadata'].get('chunk_index', ''),
                    'score': match['score']
                }
            )
            initial_docs.append(doc)

        # 再ランキング
        top_docs = rerank_documents(query, initial_docs, chat, top_k=5)

        # 出典付きテキストに変換
        documents_string = ""
        for i, doc in enumerate(top_docs, 1):
            source = doc.metadata.get("source", f"doc_{i}")
            documents_string += f"[出典{i}: {source}]\n{doc.page_content.strip()}\n\n"

        # プロンプトテンプレート
        combined_prompt = PromptTemplate(
            template="""あなたは政策提言や中小企業支援に精通したプロのアドバイザーです。

以下の検索文章をもとに、ユーザーからの質問に対して信頼できる回答をしてください。

- 回答には必ずどの「出典」に基づいているかを番号で明示してください（例：出典1）。
- 明確な根拠がない場合は「知識にないため、回答できません」と答えてください。
- 回答は丁寧で、文脈に基づいた洞察的なものにしてください。

###
検索文章:
{document}

質問:
{query}

回答:
""",
            input_variables=["document", "query"]
        )

        # Chat実行
        messages = [
            SystemMessage(content="あなたは政策や経済動向に詳しいコンサルタントです。正確な回答を心がけてください。"),
            HumanMessage(content=combined_prompt.format(document=documents_string, query=query))
        ]

        response = chat.invoke(messages)
        return response.content.strip()
        
    except Exception as e:
        return f"RAG検索中にエラーが発生しました: {str(e)}"

def perform_normal_chat(content: str) -> str:
    """通常のチャット（OpenAI API）"""
    try:
        messages = [
            SystemMessage(content="あなたは親切で知識豊富なAIアシスタントです。ユーザーの質問に丁寧に回答してください。"),
            HumanMessage(content=content)
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
            timestamp=datetime.now(),
            search_type=request.search_type
        )
        
        # AI応答の生成
        if request.search_type == "fact":
            ai_content = perform_rag_search(request.content)
        else:
            ai_content = perform_normal_chat(request.content)
        
        ai_message = MessageResponse(
            id=str(uuid.uuid4()),
            content=ai_content,
            type="ai",
            timestamp=datetime.now(),
            search_type=request.search_type
        )
        
        return ai_message
        
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
        created_at=now,
        updated_at=now
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