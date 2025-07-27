from langchain_openai import OpenAIEmbeddings
# from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone
import os
from dotenv import load_dotenv
from tqdm import tqdm

# 環境変数の読み込み
load_dotenv()

# OpenAIのAPIキーを読み込み
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# PineconeのAPIキーを読み込み
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "rag-hakusho")

# 各チャンクの最大サイズ
chunk_size = 1000
overlap_ratio = 0.25

# Embeddingモデル初期化
embedding_model = OpenAIEmbeddings(
    openai_api_key=OPENAI_API_KEY,
    model="text-embedding-3-small",
    chunk_size=chunk_size,
)

# Chroma DBの初期化（コメントアウト）
# output_db_folder = "./chroma_db"
# db = Chroma(persist_directory=output_db_folder, embedding_function=embedding_model)

# Pinecone初期化
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# テキスト分割器（長い図表用）
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=int(chunk_size * overlap_ratio)
)

# 図表単位で分割する関数
def split_by_figure(md_text: str, filename: str):
    sections = md_text.split("# 第")
    documents = []
    for i, section in enumerate(sections):
        if not section.strip():
            continue
        full_text = "# 第" + section.strip()

        # さらに長すぎる図表を分割（図表構造は維持）
        chunks = text_splitter.split_text(full_text)

        for j, chunk in enumerate(chunks):
            documents.append(Document(
                page_content=chunk,
                metadata={
                    "source": filename,
                    "figure_section": f"figure{i+1}",  # ★ 修正済み
                    "chunk_index": j
                }
            ))
    return documents

# Markdownフォルダのパス
input_folder = "./output"
md_files = [f for f in os.listdir(input_folder) if f.endswith(".md")]
print(f'処理対象のMarkdownファイル: {len(md_files)}件')

# 各Markdownファイルを処理
for md_file in tqdm(md_files, desc="Processing Markdown files", unit="file"):
    try:
        md_path = os.path.join(input_folder, md_file)
        with open(md_path, "r", encoding="utf-8") as f:
            md_text = f.read()

        documents = split_by_figure(md_text, md_file)
        
        # Pineconeにドキュメントを追加
        for doc in documents:
            # ドキュメントのエンベディングを生成
            embedding = embedding_model.embed_query(doc.page_content)
            
            # Pineconeに挿入するためのベクターを作成
            vector_id = f"{doc.metadata['source']}_{doc.metadata['figure_section']}_{doc.metadata['chunk_index']}"
            
            # ベクターをPineconeに挿入
            index.upsert(
                vectors=[{
                    "id": vector_id,
                    "values": embedding,
                    "metadata": {
                        "source": doc.metadata["source"],
                        "figure_section": doc.metadata["figure_section"],
                        "chunk_index": doc.metadata["chunk_index"],
                        "text": doc.page_content
                    }
                }]
            )
        
        # Chroma DB版（コメントアウト）
        # db.add_documents(documents)

    except Exception as e:
        print(f'❌ エラーが発生しました: {md_file} - {e}')

print(f'✅ Pineconeベクトルインデックスの作成が完了しました。インデックス名: {INDEX_NAME}')
# print(f'✅ ベクトルDBの作成が完了しました。出力先: {output_db_folder}')
