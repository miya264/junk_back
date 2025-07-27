from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.schema import SystemMessage, HumanMessage
from langchain_core.documents import Document
from pinecone import Pinecone
import os
from dotenv import load_dotenv

# 環境変数の読み込み
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "rag-hakusho")

# ベクトル埋め込みモデルの初期化
embedding_model = OpenAIEmbeddings(
    openai_api_key=OPENAI_API_KEY,
    model="text-embedding-3-small",
    chunk_size=1000,
)

# Pinecone ベクトルDBの初期化
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# LLMの初期化
chat = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model="gpt-4o-mini",  
    temperature=0.3,
)

# 再ランキング用関数
def rerank_documents(query: str, docs: list[Document], chat: ChatOpenAI, top_k: int = 5) -> list[Document]:
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
    # 文書に番号を振る
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

# ユーザーからクエリを取得
query = input("質問を入力してください: ")

# 初期の検索（多めに取得）
query_embedding = embedding_model.embed_query(query)
results = index.query(
    vector=query_embedding,
    top_k=15,
    include_metadata=True
)

# Pineconeの結果をDocumentオブジェクトに変換
initial_docs = []
for match in results['matches']:
    doc = Document(
        page_content=match['metadata']['text'],
        metadata={
            'source': match['metadata']['source'],
            'figure_section': match['metadata']['figure_section'],
            'chunk_index': match['metadata']['chunk_index'],
            'score': match['score']
        }
    )
    initial_docs.append(doc)

# 再ランキングで上位抽出
top_docs = rerank_documents(query, initial_docs, chat, top_k=5)

# 出典付きテキストに変換
documents_string = ""
for i, doc in enumerate(top_docs, 1):
    source = doc.metadata.get("source", f"doc_{i}")
    documents_string += f"[出典{i}: {source}]\n{doc.page_content.strip()}\n\n"

# プロンプトテンプレート定義
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
answer = response.content.strip()

# 出力
print("\n質問：", query)
print("回答：", answer)