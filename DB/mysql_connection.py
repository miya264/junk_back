import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from contextlib import contextmanager
import mysql.connector
from mysql.connector import Error
from dotenv import load_dotenv

# .env はローカル開発用。App Service では存在しなくてもOK（環境変数が優先）
ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=False)

def _require(key: str) -> str:
    v = os.getenv(key)
    if not v:
        raise RuntimeError(f"Missing required environment variable: {key}")
    return v

class MySQLConnection:
    """MySQLデータベース接続管理クラス"""

    def __init__(self):
        # 必須ENVの検証（ここで落ちれば原因がすぐ分かる）
        host = _require("DB_HOST")
        user = _require("DB_USER")
        password = _require("DB_PASSWORD")
        database = _require("DB_NAME")
        port = int(os.getenv("DB_PORT", "3306"))

        # SSL（Azure MySQL は必須）
        ssl_ca_path = os.getenv("DB_SSL_CA_PATH")  # 例: /home/site/wwwroot/certs/DigiCertGlobalRootG2.crt.pem
        ssl_args = {}
        if ssl_ca_path and Path(ssl_ca_path).exists():
            ssl_args = {"ssl_ca": ssl_ca_path, "ssl_disabled": False}

        self.base_config = {
            "host": host,
            "port": port,
            "user": user,
            "password": password,
            "database": database,
            "charset": "utf8mb4",
            "collation": "utf8mb4_unicode_ci",
            "autocommit": True,
            "use_unicode": True,
            "connect_timeout": 5,  # タイムアウトを短縮
            "buffered": True,
            "sql_mode": "",  # パフォーマンス最適化
            **ssl_args,
        }

        self.connection = None

        # 起動ログ（パスワード以外）。本番ログには残し過ぎないよう注意
        print(
            "MySQL config:",
            {k: v for k, v in self.base_config.items() if k not in {"password"}},
        )

    def connect(self):
        try:
            if self.connection is None or not self.connection.is_connected():
                self.connection = mysql.connector.connect(**self.base_config)
                print("MySQLデータベースに接続しました")
        except Error as e:
            print(f"MySQL接続エラー: {e}")
            raise

    def disconnect(self):
        if self.connection and self.connection.is_connected():
            self.connection.close()
            print("MySQLデータベース接続を切断しました")

    @contextmanager
    def get_connection_and_cursor(self):
        connection = None
        cursor = None
        try:
            connection = mysql.connector.connect(**self.base_config)
            cursor = connection.cursor(dictionary=True)
            yield connection, cursor
        except Error as e:
            if connection:
                connection.rollback()
            print(f"MySQL操作エラー: {e}")
            raise
        finally:
            if cursor:
                cursor.close()
            if connection and connection.is_connected():
                connection.close()

    def execute_query(self, query: str, params: Optional[Tuple] = None) -> List[Dict[str, Any]]:
        try:
            with self.get_connection_and_cursor() as (connection, cursor):
                cursor.execute(query, params or ())
                if query.strip().upper().startswith("SELECT"):
                    return cursor.fetchall()
                else:
                    connection.commit()
                    return []
        except Error as e:
            print(f"クエリ実行エラー: {e}")
            raise

    def execute_many(self, query: str, params_list: List[Tuple]) -> None:
        try:
            with self.get_connection_and_cursor() as (connection, cursor):
                cursor.executemany(query, params_list)
                connection.commit()
        except Error as e:
            print(f"一括実行エラー: {e}")
            raise

    def get_last_insert_id(self) -> int:
        try:
            with self.get_connection_and_cursor() as (connection, cursor):
                cursor.execute("SELECT LAST_INSERT_ID() as id")
                result = cursor.fetchone()
                return result["id"] if result else None
        except Error as e:
            print(f"ID取得エラー: {e}")
            raise

# グローバルインスタンス
_mysql_connection = None

def get_mysql_db() -> MySQLConnection:
    global _mysql_connection
    if _mysql_connection is None:
        _mysql_connection = MySQLConnection()
    return _mysql_connection
