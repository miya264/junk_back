#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import io
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.schema import SystemMessage, HumanMessage
from langchain_core.documents import Document
from pinecone import Pinecone

# UTF-8出力設定
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 環境変数の読み込み
load_dotenv()

def test_rag_search():
    """RAG検索をテスト"""
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

    query = "経済発展に関する政策について教えてください"
    
    try:
        print(f"クエリ: {query}")
        print("RAG検索を開始...")
        
        # ベクトル検索
        query_embedding = embedding_model.embed_query(query)
        print(f"埋め込み完了: {len(query_embedding)}次元")
        
        results = index.query(
            vector=query_embedding,
            top_k=5,
            include_metadata=True
        )
        
        print(f"検索結果: {len(results['matches'])}件")
        
        # 結果を表示
        for i, match in enumerate(results['matches'][:3]):
            print(f"\n--- 結果 {i+1} (スコア: {match['score']:.3f}) ---")
            text = match['metadata']['text'][:200] + "..."
            print(text)
            
        # 簡単なRAG応答を生成
        if results['matches']:
            context = ""
            for match in results['matches'][:3]:
                context += f"{match['metadata']['text']}\n\n"
            
            messages = [
                SystemMessage(content="あなたは政策に詳しいアシスタントです。"),
                HumanMessage(content=f"以下の情報に基づいて質問に答えてください。\n\n情報:\n{context}\n\n質問: {query}")
            ]
            
            response = chat.invoke(messages)
            print(f"\n=== RAG応答 ===")
            print(response.content)
            
    except Exception as e:
        print(f"エラー: {e}")

if __name__ == "__main__":
    test_rag_search()