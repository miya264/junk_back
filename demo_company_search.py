# demo_company_search.py
# スタンドアロン：Pineconeに会社情報を埋め込み→検索
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# 会社検索用のインデックス名（新規作成推奨）
INDEX_NAME = os.getenv("PINECONE_COMPANY_INDEX_NAME", "company-info").strip()

# Serverlessの作成先（ragと揃えるなど環境に合わせて）
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")

EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536  # 上記モデルの次元

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY が未設定です (.env を確認)")
if not PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY が未設定です (.env を確認)")

# ---- 元データ（タプル形式）
companies_raw = [
    ('トヨタ自動車株式会社', '1180301018771', '471-8571', '愛知県豊田市トヨタ町1番地', '1937-08-28', 635504000000, 375235, '自動車の生産、販売、リース、部品製造、住宅製造など。'),
    ('ソニーグループ株式会社', '5010401067252', '105-8664', '東京都港区港南1-7-1', '1946-05-07', 848981452500, 113000, 'エレクトロニクス製品の開発、製造、販売、音楽、映画、金融など多岐にわたる事業。'),
    ('任天堂株式会社', '1130001011420', '601-8501', '京都府京都市南区上鳥羽鉾立町11-1', '1947-11-20', 10065400000, 7552, '家庭用レジャー機器、コンピュータゲームの開発、製造、販売など。'),
    ('パナソニックホールディングス株式会社', '5120001158218', '571-8501', '大阪府門真市大字門真1006番地', '1935-12-15', 259384000000, 233391, '家電、住宅設備、車載、B2Bソリューションなど。'),
    ('株式会社日立製作所', '7010001008844', '100-8280', '東京都千代田区丸の内1-6-6', '1920-02-01', 457813000000, 295982, 'デジタルシステム＆サービス、グリーンエナジー＆モビリティ、コネクティブインダストリーズなど。'),
    ('ソフトバンク株式会社', '9010401052465', '105-7529', '東京都港区海岸1-7-1', '1986-12-09', 228000000000, 20387, '移動通信サービス、固定通信サービス、IoT、AI、ロボットなど。'),
    ('楽天グループ株式会社', '9010701020592', '158-0094', '東京都世田谷区玉川1-14-1', '1997-02-07', 416599000000, 30129, 'ECサイト「楽天市場」、金融、通信、メディア、プロスポーツなど。'),
    ('株式会社ブリヂストン', '3010001034943', '104-8340', '東京都中央区京橋3-1-1', '1931-03-01', 126354000000, 137839, 'タイヤ、工業製品、スポーツ用品などの製造・販売。'),
    ('日本電気株式会社', '7010401022916', '108-8001', '東京都港区芝5-7-1', '1899-07-17', 426549000000, 20958, '通信機器、コンピュータシステム、ITソリューション、電子部品など。'),
    ('株式会社ニトリホールディングス', '5430001012905', '060-0061', '北海道札幌市中央区南一条西8丁目1-2', '1972-03-03', 14204000000, 34390, '家具・インテリア用品の企画・製造・物流・販売。'),
    ('旭事務器株式会社', '1010001000030', '101-0047', '東京都千代田区内神田3-22-6', '1961-07-19', 20000000, 10, '事務機器、OA機器販売。'),
    ('朝日テック工業株式会社', '1010001000039', '120-0026', '東京都足立区千住旭町42-6', '1960-04-18', 21600000, 10, 'ろ布、バグフィルターなどの製造・販売。'),
    ('アサヒプランテック株式会社', '1010001000047', '140-0014', '東京都品川区大井4-13-14', '1965-03-01', 10000000, 18, '機械、プラント設備の設計・製作。'),
    ('株式会社アイ・アイ・エム', '1010001000055', '143-0016', '東京都大田区大森北1-23-2', '1973-10-01', 30000000, 60, '性能管理ツール「ES/1 NEO」の開発・販売。'),
    ('アイ・アール・プランニング株式会社', '1010001000063', '104-0045', '東京都中央区築地6-16-1', '1995-12-08', None, 12, '施設管理、不動産、飲食事業など。'),
    ('伊藤時商事株式会社', '1010001000666', '101-0047', '東京都千代田区内神田3-22-6', None, None, None, '情報なし'),
    ('株式会社アイエスエイ・ジャパン', '1010001000071', '105-0014', '東京都港区芝1-5-14', '1970-02-12', 30000000, None, 'コンピュータシステム販売、ソフトウェア開発など。'),
    ('会津工業株式会社', '1010001000138', '965-0004', '福島県会津若松市一箕町大字八幡字下島16-1', '1969-02-01', None, 65, '土木・建設事業、アスファルト舗装など。'),
    ('愛電株式会社', '1010001000146', '105-0012', '東京都港区芝大門1-4-10', '1972-01-20', None, None, '電気設備工事、空調設備工事、リフォームなど。'),
    ('株式会社アートランド', '1010001000170', '102-0083', '東京都千代田区麹町3-5-1', '2001-08-01', 10000000, None, '笑いヨガ、ストレスマネジメント関連事業。'),
    ('エイコートレーディング株式会社', '1010001000996', '105-0021', '東京都港区東新橋1-1-21', None, None, None, '情報なし'),
    ('株式会社アトラスビル', '1010001000402', '102-0071', '東京都千代田区富士見1-4-11', None, None, None, 'ビル総合管理、清掃、設備点検など。'),
    ('エトワール株式会社', '1010001001210', '104-0061', '東京都中央区銀座1-19-14', None, None, None, '婦人服、雑貨の企画・販売。'),
    ('株式会社エヌワイ弘法', '1010001001251', '101-0046', '東京都千代田区神田多町2-1', '1990-07-26', None, None, 'オフィスマネジメント、オフィスコンサルティングなど。'),
    ('株式会社エフピー研究所', '1010001001276', '102-0074', '東京都千代田区九段南3-8-1', '1990-04-02', 10000000, 10, 'ファイナンシャルプランニングソフトの開発・販売。'),
    ('株式会社パルームシティ', '1010001001284', '105-0012', '東京都港区芝大門1-4-10', None, None, None, '不動産売買、仲介、賃貸管理など。'),
    ('株式会社エミネント', '1010001001318', '110-0015', '東京都台東区東上野3-3-1', '1960-03-01', 10000000, 20, '縫製品の企画・製造・販売。'),
    ('大崎工業株式会社', '1010001001425', '141-0032', '東京都品川区大崎1-11-2', None, None, None, '情報なし'),
    ('大田機材株式会社', '1010001001458', '144-0045', '東京都大田区南六郷3-2-2', None, None, None, '建築材料の卸売。'),
    ('株式会社大西設計', '1010001001482', '101-0062', '東京都千代田区神田駿河台2-2-12', None, None, None, '情報なし'),
    ('オリムピック開発株式会社', '1010001001664', '105-0004', '東京都港区新橋5-23-1', None, None, None, 'ゴルフ場、ホテル、飲食店の経営。'),
]

def tuple_to_dict(t):
    return {
        "name": t[0],
        "corporate_number": t[1],
        "postal_code": t[2],
        "location": t[3],
        "founding_date": t[4],
        "capital": t[5],
        "employee_number": t[6],
        "business_summary": t[7],
    }

companies = [tuple_to_dict(t) for t in companies_raw]

def safe(v):  # テキスト結合用の安全化
    return "" if v is None else str(v)

def safe_meta(v):  # Pinecone metadata 制約に合わせる
    if v is None:
        return ""              # Noneは空文字へ
    if isinstance(v, (int, float, bool, str)):
        return v
    return str(v)

# ---- Embeddings（名前は "embedding" に統一）
embedding = OpenAIEmbeddings(
    openai_api_key=OPENAI_API_KEY,
    model=EMBED_MODEL
)

# ---- Pinecone 初期化 & インデックス検証/作成
pc = Pinecone(api_key=PINECONE_API_KEY)

def ensure_index():
    # v2 SDK: list_indexes() は {"indexes":[{name, dimension, ...}, ...]} を返す
    existing = {i["name"]: i for i in pc.list_indexes().get("indexes", [])}
    if INDEX_NAME in existing:
        dim = existing[INDEX_NAME]["dimension"]
        if dim != EMBED_DIM:
            raise SystemExit(
                f"[ERROR] 既存インデックス '{INDEX_NAME}' の次元は {dim} です。\n"
                f"現在の埋め込みモデル({EMBED_MODEL})は {EMBED_DIM} 次元で不一致です。\n"
                f"対処: 新しいインデックス名（例: company-info）を設定して再実行、"
                f"または既存を削除して {EMBED_DIM} 次元で作り直してください。"
            )
        print(f"[OK] 既存インデックス '{INDEX_NAME}' (dim={dim}) を使用します。")
    else:
        print(f"[INFO] インデックス '{INDEX_NAME}' が無いので作成します (dim={EMBED_DIM})...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBED_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION)
        )
        print("[OK] 作成完了。")

ensure_index()
index = pc.Index(INDEX_NAME)

def reindex():
    vectors = []
    for c in companies:
        text = f"{safe(c['name'])} {safe(c['location'])} {safe(c['business_summary'])}"
        vec = embedding.embed_query(text)
        vectors.append({
            "id": safe(c["corporate_number"]) or safe(c["name"]),
            "values": vec,
            "metadata": {
                "name": safe_meta(c["name"]),
                "corporate_number": safe_meta(c["corporate_number"]),
                "location": safe_meta(c["location"]),
                "business_summary": safe_meta(c["business_summary"]),
                "postal_code": safe_meta(c["postal_code"]),
                "employee_number": safe_meta(c["employee_number"]),
                "capital": safe_meta(c.get("capital")),
                "founding_date": safe_meta(c.get("founding_date")),
            }
        })
    index.upsert(vectors=vectors)
    print(f"✅ upsert: {len(vectors)} 件")

def search(query: str, top_k: int = 5):
    qv = embedding.embed_query(query)
    res = index.query(vector=qv, top_k=top_k, include_metadata=True)

    print(f"\n🔍 Query: {query}")
    if not res.matches:
        print("  ※該当なし")
        return

    for m in res.matches:
        md = m.metadata or {}
        print(f"- {md.get('name')} ({md.get('location')})  score={getattr(m,'score',0.0):.3f}")
        print(f"  概要: {md.get('business_summary')}")


if __name__ == "__main__":
    reindex()
    search("ゲーム関連の会社")
    search("自動車メーカー")
