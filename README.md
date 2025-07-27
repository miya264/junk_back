# AI Agent Backend

FastAPIを使用したAIエージェントのバックエンドAPIです。

## 機能

- **通常チャット**: OpenAI APIを使用した一般的なチャット機能
- **ファクト検索**: RAG（Retrieval-Augmented Generation）を使用した事実検索機能
- **人脈検索**: ネットワーク検索機能（現在は通常チャットと同じ）

## セットアップ

### 1. 依存関係のインストール

```bash
pip install -r requirements.txt
```

### 2. 環境変数の設定

`env.example`をコピーして`.env`ファイルを作成し、必要なAPIキーを設定してください：

```bash
cp env.example .env
```

`.env`ファイルを編集して以下を設定：

```env
OPENAI_API_KEY=your_openai_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_INDEX_NAME=rag-hakusho
```

### 3. サーバーの起動

```bash
python main.py
```

または

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## API エンドポイント

### POST /api/chat

チャットメッセージを送信します。

**リクエストボディ:**
```json
{
  "content": "質問内容",
  "search_type": "normal"  // "normal", "fact", "network"
}
```

**レスポンス:**
```json
{
  "id": "uuid",
  "content": "AIの回答",
  "type": "ai",
  "timestamp": "2024-01-01T00:00:00",
  "search_type": "normal"
}
```

### GET /api/sessions

チャットセッション一覧を取得します。

### POST /api/sessions

新しいチャットセッションを作成します。

## 使用方法

1. サーバーを起動
2. フロントエンドからAPIにリクエストを送信
3. `search_type`パラメータで検索タイプを指定：
   - `"normal"`: 通常のチャット
   - `"fact"`: RAG検索
   - `"network"`: 人脈検索（現在は通常チャットと同じ）

## 開発

### 開発サーバーの起動

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### API ドキュメント

サーバー起動後、以下のURLでAPIドキュメントを確認できます：

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc 