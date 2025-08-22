# MySQL版 AI Agent API

Azure MySQLを使用したAI Agent APIのセットアップ手順です。

## 前提条件

- Python 3.8以上
- Azure MySQL Database
- 必要な環境変数が設定済み

## 環境変数の設定

`.env`ファイルに以下の環境変数を設定してください：

```env
# OpenAI API
OPENAI_API_KEY=your_openai_api_key

# Pinecone
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=rag-hakusho

# MySQL Database
DB_HOST=your_mysql_host.mysql.database.azure.com
DB_PORT=3306
DB_USER=your_mysql_username
DB_PASSWORD=your_mysql_password
DB_NAME=your_database_name
```

## インストール

1. 依存関係をインストール：
```bash
pip install -r requirements.txt
```

2. MySQLデータベースを作成・初期化：
```bash
python create_mysql_db.py
```

## 使用方法

### 1. データベース初期化

```bash
python create_mysql_db.py
```

このスクリプトは以下を実行します：
- データベースの存在確認
- 必要に応じてデータベースの作成
- テーブルスキーマの作成
- サンプルデータの挿入

### 2. データベース内容確認

```bash
python query_mysql_db.py
```

このスクリプトは以下を表示します：
- 各テーブルの最新データ
- データベース統計情報

### 3. APIサーバーの起動

```bash
# MySQL版APIサーバーを起動
python main_mysql.py

# または
uvicorn main_mysql:app --host 0.0.0.0 --port 8000
```

## ファイル構成

### MySQL用ファイル
- `mysql_connection.py` - MySQL接続管理クラス
- `mysql_crud.py` - MySQL用CRUD操作
- `create_mysql_database.sql` - MySQL用データベーススキーマ
- `create_mysql_db.py` - データベース作成・初期化スクリプト
- `query_mysql_db.py` - データベース内容確認スクリプト
- `main_mysql.py` - MySQL版FastAPIアプリケーション

### 共通ファイル
- `policy_agents.py` - 政策立案エージェント
- `flexible_policy_agents.py` - 柔軟な政策立案エージェント
- `requirements.txt` - 依存関係

## データベーススキーマ

### 主要テーブル
- `authentication` - 認証情報
- `coworkers` - 同僚情報
- `projects` - プロジェクト情報
- `project_members` - プロジェクトメンバー
- `policy_steps` - 政策ステップ
- `project_step_sections` - プロジェクトステップセクション
- `chat_sessions` - チャットセッション
- `chat_messages` - チャットメッセージ
- `rag_search_results` - RAG検索結果
- `audit_logs` - 監査ログ

## API エンドポイント

### チャット関連
- `POST /api/chat` - チャットメッセージ送信
- `POST /api/policy-flexible` - 柔軟な政策立案
- `GET /api/session-state/{session_id}` - セッション状態取得

### プロジェクト関連
- `POST /api/projects` - プロジェクト作成
- `GET /api/projects/{project_id}` - プロジェクト詳細取得
- `GET /api/projects/by-coworker/{coworker_id}` - 同僚のプロジェクト一覧

### ステップセクション関連
- `POST /api/project-step-sections` - ステップセクション保存
- `GET /api/project-step-sections/{project_id}/{step_key}` - ステップセクション取得

### 同僚関連
- `GET /api/coworkers/search` - 同僚検索

## 主な機能

### 1. RAG検索
- ファクト検索ボタンでRAGが起動
- 検索クエリと結果がデータベースに保存
- 出典情報も含めて保存

### 2. チャット機能
- 通常チャット
- 政策立案ステップ別チャット
- セッション管理

### 3. プロジェクト管理
- プロジェクト作成・管理
- メンバー管理
- ステップ別セクション管理

## トラブルシューティング

### 接続エラー
1. 環境変数が正しく設定されているか確認
2. Azure MySQLのファイアウォール設定を確認
3. SSL接続が必要な場合は適切に設定

### データベースエラー
1. データベースが存在するか確認
2. ユーザー権限を確認
3. テーブルが正しく作成されているか確認

### 依存関係エラー
```bash
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

## 開発・テスト

### データベースリセット
```bash
python create_mysql_db.py
# 既存データベースの削除確認で 'y' を入力
```

### ログ確認
```bash
# APIサーバーのログを確認
python main_mysql.py
```

## 本番環境での注意点

1. **セキュリティ**
   - 環境変数の適切な管理
   - データベース接続の暗号化
   - ファイアウォール設定

2. **パフォーマンス**
   - データベース接続プールの設定
   - インデックスの最適化
   - クエリの最適化

3. **バックアップ**
   - 定期的なデータベースバックアップ
   - ログファイルの管理

## サポート

問題が発生した場合は、以下を確認してください：
1. ログファイルの確認
2. データベース接続の確認
3. 環境変数の設定確認
