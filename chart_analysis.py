import os
from tqdm import tqdm
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv

# === 環境変数の読み込み ===
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AZURE_FORM_KEY = os.getenv("AZURE_FORM_KEY")
AZURE_FORM_ENDPOINT = os.getenv("AZURE_FORM_ENDPOINT")

# === クライアントの初期化 ===
form_client = DocumentAnalysisClient(
    endpoint=AZURE_FORM_ENDPOINT,
    credential=AzureKeyCredential(AZURE_FORM_KEY)
)
llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model="gpt-4o-mini", 
    temperature=0.7
)

# === ローカルPDFからテキストを抽出する関数 ===
def analyze_pdf_page_layout(file_path: str) -> str:
    with open(file_path, "rb") as f:
        poller = form_client.begin_analyze_document(
            model_id="prebuilt-layout",
            document=f
        )
        result = poller.result()
    page = result.pages[0]
    lines = [line.content for line in page.lines]
    return "\n".join(lines)

# === 図表だけ抽出するLLMプロンプト ===
def analyze_chart_from_page(text: str, page_num: int, llm: ChatOpenAI) -> str:
    prompt = [
        SystemMessage(content=(
            "あなたは統計図表を読み取り、政策文書に適した分析をMarkdown形式で記述する専門家です。\n"
            "以下のテキストから図表部分を抽出し、それぞれの図表タイトルに対して：\n"
            "- 何を示すか\n"
            "- 傾向や注目点\n"
            "- 数値や文脈の解釈\n"
            "を丁寧に記述してください。"
        )),
        HumanMessage(content=f'## ページ{page_num}\n\n{text}')
    ]
    return llm.invoke(prompt).content

# === メイン処理 ===
def reextract_charts_only(split_dir: str, llm: ChatOpenAI, output_path: str):
    chart_results = []
    files = sorted([f for f in os.listdir(split_dir) if f.endswith(".pdf")])

    for i, fname in enumerate(tqdm(files, desc="図表のみ再抽出中")):
        full_path = os.path.join(split_dir, fname)
        raw_text = analyze_pdf_page_layout(full_path)
        chart_md = analyze_chart_from_page(raw_text, i + 1, llm)
        chart_results.append(chart_md)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(chart_results))

    print(f"✅ 図表出力完了: {output_path}")

# === 実行例 ===
if __name__ == "__main__":
    reextract_charts_only(
        split_dir="output/split_pages",
        llm=llm,
        output_path="output/charts_only_reextract.md"
    )

