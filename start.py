#!/usr/bin/env python3
"""
AI Agent Backend Server Startup Script
"""

import os
import sys
from dotenv import load_dotenv

# 環境変数を読み込み
load_dotenv()

def check_environment():
    """環境変数のチェック"""
    required_vars = ['OPENAI_API_KEY', 'PINECONE_API_KEY']
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print("❌ 以下の環境変数が設定されていません:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\n.envファイルを確認してください。")
        return False
    
    print("✅ 環境変数の設定を確認しました")
    return True

def main():
    """メイン関数"""
    print("🚀 AI Agent Backend Server を起動しています...")
    
    # 環境変数チェック
    if not check_environment():
        sys.exit(1)
    
    try:
        import uvicorn
        from main import app
        
        print("📡 サーバーを起動中...")
        print("🌐 API ドキュメント: http://localhost:8000/docs")
        print("🔗 サーバーURL: http://localhost:8000")
        print("⏹️  停止するには Ctrl+C を押してください")
        print("-" * 50)
        
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
        
    except ImportError as e:
        print(f"❌ 必要なパッケージがインストールされていません: {e}")
        print("以下のコマンドを実行してください:")
        print("pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"❌ サーバーの起動に失敗しました: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 