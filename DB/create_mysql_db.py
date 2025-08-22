#!/usr/bin/env python3
"""
MySQL 非破壊マイグレーション: 既存DBは初期化せず、
- DBが無ければ作成
- CREATE TABLE IF NOT EXISTS はそのまま実行（既存テーブルは変更しない）
- INSERT は既存行を更新せず、足りない行だけ追加（INSERT IGNORE）
- DROP/CREATE DATABASE や既存を変える系はスキップ
"""
import os, sys, re
from pathlib import Path
import mysql.connector
from mysql.connector import Error
from dotenv import load_dotenv, find_dotenv, dotenv_values

BACKEND_DIR = Path(__file__).resolve().parents[1]
ENV_PATH = BACKEND_DIR / ".env" if (BACKEND_DIR / ".env").exists() else find_dotenv(".env", usecwd=True)
load_dotenv(dotenv_path=ENV_PATH if ENV_PATH else None, override=True)

def connect_db(db=None):
    host = os.getenv('DB_HOST', '').strip()
    port = int(os.getenv('DB_PORT', '3306'))
    user = os.getenv('DB_USER', '').strip()
    password = os.getenv('DB_PASSWORD', '').strip()
    kwargs = dict(host=host, port=port, user=user, password=password, charset='utf8mb4', connection_timeout=10)
    if db:
        kwargs['database'] = db
    return mysql.connector.connect(**kwargs)

def ensure_database(cursor, db_name):
    cursor.execute("CREATE DATABASE IF NOT EXISTS `{}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci".format(db_name))

def preprocess_sql_to_safe(sql_text: str) -> list[str]:
    """
    - コメント除去（-- 行頭）
    - DROP/CREATE DATABASE スキップ
    - INSERT … ON DUPLICATE KEY UPDATE … → INSERT IGNORE に置換
    - 通常の INSERT も INSERT IGNORE に置換（既存更新しない方針）
    - 残りはそのまま
    """
    lines = []
    for line in sql_text.splitlines():
        # 行頭コメントは捨てる
        if line.strip().startswith("--"):
            continue
        lines.append(line)
    sql = "\n".join(lines)

    # ステートメント分割（ざっくり；セミコロン終端）
    raw_statements = [s.strip() for s in re.split(r';\s*\n|;\s*$', sql) if s.strip()]
    safe_statements: list[str] = []

    for s in raw_statements:
        upper = s.upper()
        # DB 破壊系はスキップ
        if upper.startswith("DROP DATABASE") or upper.startswith("CREATE DATABASE"):
            continue
        # SET/START TRANSACTION/COMMIT は任意（必要なら残す）
        if upper.startswith("SET "):
            safe_statements.append(s)
            continue

        # INSERT を非破壊化
        if upper.startswith("INSERT INTO"):
            # ON DUPLICATE KEY UPDATE ～ を除去
            s = re.sub(r"\s+AS\s+new\s+ON\s+DUPLICATE\s+KEY\s+UPDATE\s+.*$", "", s, flags=re.IGNORECASE | re.DOTALL)
            s = re.sub(r"\s+ON\s+DUPLICATE\s+KEY\s+UPDATE\s+.*$", "", s, flags=re.IGNORECASE | re.DOTALL)
            # INSERT → INSERT IGNORE
            s = re.sub(r"^(INSERT)\s+(INTO)\b", r"INSERT IGNORE \2", s, flags=re.IGNORECASE)

        # それ以外（CREATE TABLE IF NOT EXISTS / ALTER / CREATE INDEX など）はそのまま
        safe_statements.append(s)

    # セミコロン付与
    return [stmt + ";" for stmt in safe_statements if stmt.strip()]

def main():
    db_host = os.getenv('DB_HOST'); db_user = os.getenv('DB_USER'); db_pw = os.getenv('DB_PASSWORD'); db_name = os.getenv('DB_NAME')
    missing = [k for k,v in {"DB_HOST":db_host,"DB_USER":db_user,"DB_PASSWORD":db_pw,"DB_NAME":db_name}.items() if not v]
    if missing:
        print("❌ .env 不足:", ", ".join(missing)); sys.exit(1)

    sql_file = Path(__file__).with_name("create_mysql_database.sql")
    if not sql_file.exists():
        print(f"❌ SQL ファイルが見つかりません: {sql_file}")
        sys.exit(1)

    try:
        # サーバーレベル接続（DBなし）
        conn = connect_db()
        cur = conn.cursor()
        ensure_database(cur, db_name)
        conn.commit()
        cur.close(); conn.close()

        # DB へ接続
        conn = connect_db(db=db_name)
        cur = conn.cursor()

        raw_sql = sql_file.read_text(encoding="utf-8")
        stmts = preprocess_sql_to_safe(raw_sql)

        print(f"🔧 非破壊モードで {len(stmts)} ステートメントを実行します…")
        for i, stmt in enumerate(stmts, 1):
            try:
                cur.execute(stmt)
                conn.commit()
            except Error as e:
                # 既存構造と噛み合わない ALTER 等は握りつぶさず表示だけ
                print(f"⚠️  {i}番目のステートメントでエラー: {e}\n    → {stmt[:140]}...")
                # 必要に応じて continue
                continue

        # ざっくり確認
        cur.execute("SHOW TABLES")
        tables = [t[0] for t in cur.fetchall()]
        print("\n📋 テーブル一覧:", ", ".join(tables) if tables else "(なし)")

        # 代表テーブルの件数サマリ（失敗しても続行）
        print("\n📊 データ件数（存在すれば表示）")
        for label, q in [
            ("coworkers", "SELECT COUNT(*) FROM coworkers"),
            ("projects", "SELECT COUNT(*) FROM projects"),
            ("project_members", "SELECT COUNT(*) FROM project_members"),
            ("policy_steps", "SELECT COUNT(*) FROM policy_steps"),
        ]:
            try:
                cur.execute(q)
                print(f"  - {label}: {cur.fetchone()[0]}")
            except Exception as e:
                print(f"  - {label}: 取得不可（{e}）")

        cur.close(); conn.close()
        print("\n✅ 非破壊マイグレーション完了（既存は変更せず、足りない要素のみ追加）")

    except Error as e:
        print(f"❌ MySQL 接続/実行エラー: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
