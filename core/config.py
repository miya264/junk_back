import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# .envファイルを読み込む
ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=False)

# 環境変数
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "rag-hakusho")
GBIZINFO_API_KEY = os.getenv("GBIZINFO_API_KEY")
GBIZINFO_URL = os.getenv("GBIZINFO_URL", "https://info.gbiz.go.jp/hojin/v1/hojin")

# AIモデルの初期化
embedding_model = None
chat = None

def init_models():
    """AIモデルの初期化と検証"""
    global embedding_model, chat
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
    else:
        print("⚠️ Warning: OPENAI_API_KEY is not set. AI models will not be available.")
    return embedding_model, chat