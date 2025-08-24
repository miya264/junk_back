import fitz
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv
import os
from typing import Optional
from tqdm import tqdm
import time

# 環境変数の読み込み
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AZURE_FORM_KEY = os.getenv("AZURE_FORM_KEY")
AZURE_FORM_ENDPOINT = os.getenv("AZURE_FORM_ENDPOINT")

form_client = DocumentAnalysisClient(
    endpoint=AZURE_FORM_ENDPOINT,
    credential=AzureKeyCredential(AZURE_FORM_KEY)
)

llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model="gpt-4o-mini", 
    temperature=0.7
)

OUTPUT_DIR = "output"
SPLIT_DIR = os.path.join(OUTPUT_DIR, "split_pages")
os.makedirs(SPLIT_DIR, exist_ok=True)

def split_pdf_per_page(pdf_path: str) -> list:
    doc = fitz.open(pdf_path)
    split_paths = []
    for i in range(len(doc)):
        new_doc = fitz.open()
        new_doc.insert_pdf(doc, from_page=i, to_page=i)
        split_path = os.path.join(SPLIT_DIR, f"page_{i + 1}.pdf")
        new_doc.save(split_path)
        new_doc.close()
        split_paths.append(split_path)
    return split_paths

def analyze_pdf_page_local(file_path: str, page_num: int, max_retries: int = 3, retry_delay: float = 2):
    for attempt in range(max_retries):
        try:
            with open(file_path, "rb") as f:
                poller = form_client.begin_analyze_document(
                    model_id="prebuilt-layout",
                    document=f
                )
                result = poller.result()

            page = result.pages[0]
            lines = [line.content for line in page.lines]
            extracted_text = "\n".join(lines)

            messages = [
                SystemMessage(content=(
                    "あなたは優秀な政策や法案の内容を分析し、わかりやすくまとめる専門家です。\n\n"
                    "以下のテキストは、政策や法案の内容を抽出したものです。\n\n"
                    "あなたの仕事は、このテキストをわかりやすくまとめ、Markdown形式で出力してください。\n\n"
                    "句読点や改行の整備、表や項目の整理も行ってください。"
                )),
                HumanMessage(content=f'## ページ{page_num}\n\n###抽出されたテキスト\n\n{extracted_text}')
            ]

            response = llm.invoke(messages)
            return extracted_text, response.content

        except Exception as e:
            print(f"Error on page {page_num}, attempt {attempt + 1}: {e}")
            if attempt == max_retries - 1:
                return "", f"【整形失敗】ページ{page_num}"
            time.sleep(retry_delay)

def extract_and_restructure_pdf(pdf_path: str, start_page: int = 1) -> None:
    print("📄 全ページを分割して処理します...")
    all_pages = split_pdf_per_page(pdf_path)
    pdf_pages = all_pages[start_page - 1:]

    raw_texts = []
    markdown_results = []

    for i, page_path in enumerate(tqdm(pdf_pages, desc="PDFページ処理中")):
        page_num = start_page + i
        extracted_text, markdown = analyze_pdf_page_local(page_path, page_num)
        raw_texts.append(f"## ページ{page_num}\n{extracted_text}")
        markdown_results.append(markdown)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "output.txt"), 'w', encoding='utf-8') as raw_file:
        raw_file.write('\n\n'.join(raw_texts))
    with open(os.path.join(OUTPUT_DIR, "output.md"), 'w', encoding='utf-8') as md_file:
        md_file.write('\n\n'.join(markdown_results))

    print("✅ 処理完了！output/output.txt と output/output.md を確認してください。")

def process_all_pdfs_in_folder(folder_path: str):
    """
    指定フォルダ内の全PDFを一括で処理し、output/配下にファイルごとにテキスト・Markdownを保存する
    """
    pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]
    if not pdf_files:
        print("PDFファイルが見つかりませんでした。")
        return

    for pdf_file in pdf_files:
        pdf_path = os.path.join(folder_path, pdf_file)
        print(f"\n=== {pdf_file} の処理を開始 ===")
        all_pages = split_pdf_per_page(pdf_path)
        raw_texts = []
        markdown_results = []
        for i, page_path in enumerate(tqdm(all_pages, desc=f"{pdf_file}のページ処理中")):
            page_num = i + 1
            extracted_text, markdown = analyze_pdf_page_local(page_path, page_num)
            raw_texts.append(f"## ページ{page_num}\n{extracted_text}")
            markdown_results.append(markdown)
        # ファイル名ベースで保存
        base_name = os.path.splitext(pdf_file)[0]
        with open(os.path.join(OUTPUT_DIR, f"{base_name}.txt"), 'w', encoding='utf-8') as raw_file:
            raw_file.write('\n\n'.join(raw_texts))
        with open(os.path.join(OUTPUT_DIR, f"{base_name}.md"), 'w', encoding='utf-8') as md_file:
            md_file.write('\n\n'.join(markdown_results))
        print(f"✅ {pdf_file} の処理完了！output/{base_name}.txt, output/{base_name}.md を確認してください。\n")

if __name__ == "__main__":
    # process_all_pdfs_in_folderを呼び出す
    process_all_pdfs_in_folder("text")






#OPENAIのみ
# def extract_and_restructure_pdf(
#         pdf_path: str,
#         output_md_path: str,
#         output_raw_text_path: str,
#         first_page:Optional[int] = 1,
#         last_page:Optional[int] = None,
#         max_retries:int = 3,
#         retry_delay:float = 2

# ) -> None:
#     """
#     PDFファイルを読み込み、テキストを抽出して、Markdown形式で保存する

#      Args:
#             pdf_path (str): 処理するPDFファイルのパス。
#             output_md_path (str): 整形結果を保存するMarkdownファイルのパス。
#             output_raw_text_path (str): 抽出された生テキストを保存するファイルのパス。
#             first_page (Optional[int]): 処理を開始するページ番号（1始まり）。デフォルトは1。
#             last_page (Optional[int]): 処理を終了するページ番号。デフォルトは最終ページ。
#             max_retries (int): LLM呼び出しの最大リトライ回数。デフォルトは3。
#             retry_delay (int): LLM呼び出し失敗時のリトライ間隔（秒）。デフォルトは2秒。

#     """
#     #pdfを開く
#     doc = fitz.open(pdf_path)
#     total_pages = len(doc)

#     #ページ範囲の計算
#     first_page = first_page or 1
#     last_page = last_page or total_pages

#     raw_texts = [] #生テキストを保存するリスト
#     markdown_results = [] #整形後のテキストを保存するリスト

#     #PDFページを順に処理
#     for page_num in tqdm(range(first_page - 1, last_page), desc="Processing PDF pages"):
#         page = doc[page_num]
#         extracted_text = page.get_text("text") #テキストの抽出

#         #LLMへの指示を作成
#         messages = [
#             SystemMessage(content=(
#                 "あなたは優秀な政策や法案の内容を分析し、わかりやすくまとめる専門家です。\n\n"
#                 "以下のテキストは、政策や法案の内容を抽出したものです。\n\n"
#                 "あなたの仕事は、このテキストをわかりやすくまとめ、Markdown形式で出力することです。\n\n"
#                 "出力する際は、以下のルールに従ってください。\n\n"
#                 "1.句読点や改行の位置を適切に整え、誤字脱字を修正してください（文脈に基づく範囲内で）。\n"
#                 "2. 元のテキストに含まれる情報を削除しないでください。\n"
#                 "3. 表形式のデータは可能な限り元のレイアウトを維持してください。\n"
#                 "4. グラフの軸の数値関係を確認し、適切に説明してください。\n\n"
#                 "最終結果はMarkdown形式で出力してください。"
#             )),

#             HumanMessage(content=f'## ページ{page_num + 1}\n\n###抽出されたテキスト\n\n{extracted_text}')
#         ]

#         #LLMへの呼び出しを試行
#         for attempt in range(max_retries):
#             try:
#                 response = llm.invoke(messages)
#                 markdown_results.append(response.content)
#                 break
#             except Exception as e:
#                 print(f"Error during OpenAI API call on page {page_num + 1} (attempt {attempt + 1}): {e}")
#                 if attempt == max_retries - 1:
#                     print(f'Failed after max retries for page {page_num + 1}')
#                 time.sleep(retry_delay)

#         #抽出されたテキストを保存
#         raw_texts.append(f'## ページ{page_num + 1}\n{extracted_text}')

#     #生テキストをファイルに保存
#     with open(output_raw_text_path, 'w', encoding='utf-8') as raw_file:
#         raw_file.write('\n\n'.join(raw_texts))
#     print(f'抽出されたテキストが保存されました: {output_raw_text_path}')

#     #整形結果をMarkdownで保存
#     with open(output_md_path, 'w', encoding='utf-8') as md_file:
#         md_file.write('\n\n'.join(markdown_results))
#     print(f'整形結果がMarkdownファイルに保存されました: {output_md_path}')

# if __name__ == "__main__":
#     #使用例
#     pdf_path = "03Hakusyo_part1_chap1_web.pdf" #処理するPDFファイルのパス
#     output_md_path = "output.md" #整形結果の保存先
#     output_raw_text_path = "output.txt" #抽出されたテキストの保存先

#     #出力ディレクトリの作成
#     os.makedirs('output', exist_ok=True)

#     #PDF処理の実行
#     extract_and_restructure_pdf(
#         pdf_path=pdf_path,
#         output_md_path=output_md_path,
#         output_raw_text_path=output_raw_text_path,
#         first_page=None,
#         last_page=None
# )
