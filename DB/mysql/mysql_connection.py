import os
import ssl
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import pymysql
from dotenv import load_dotenv

ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=False)

def _require(key: str) -> str:
    v = os.getenv(key)
    if not v:
        raise RuntimeError(f"Missing required environment variable: {key}")
    return v

class MySQLConnection:
    """MySQLデータベース接続管理クラス (PyMySQL使用)"""

    def __init__(self):
        host = _require("DB_HOST")
        user = _require("DB_USER")
        password = _require("DB_PASSWORD")
        database = _require("DB_NAME")
        port = int(os.getenv("DB_PORT", "3306"))
        
        # SSL configuration using PyMySQL
        ssl_args = {'verify_mode': ssl.CERT_NONE}
        print("SSL enabled with PyMySQL")

        self.base_config = {
            "host": host,
            "port": port,
            "user": user,
            "password": password,
            "database": database,
            "charset": "utf8mb4",
            "autocommit": True,
            "ssl": ssl_args,
        }
        self.connection_ready = False

    async def initialize_pool(self):
        """接続設定を初期化"""
        try:
            # Test connection
            conn = pymysql.connect(**self.base_config)
            cursor = conn.cursor(pymysql.cursors.DictCursor)
            cursor.execute("SELECT 1")
            cursor.fetchone()
            cursor.close()
            conn.close()
            
            self.connection_ready = True
            print("MySQL connection initialized successfully")
            return True
        except Exception as e:
            print(f"MySQL接続初期化エラー: {e}")
            self.connection_ready = False
            return False

    def get_connection_and_cursor(self):
        """接続とカーソルを取得する同期関数"""
        conn = None
        cursor = None
        try:
            conn = pymysql.connect(**self.base_config)
            cursor = conn.cursor(pymysql.cursors.DictCursor)
            return conn, cursor
        except Exception as e:
            if cursor:
                cursor.close()
            if conn:
                conn.close()
            print(f"MySQL接続エラー: {e}")
            raise

    async def execute_query(self, query: str, params: Optional[Tuple] = None) -> List[Dict[str, Any]]:
        if not self.connection_ready:
            raise RuntimeError("Database connection is not initialized. Database operations are not available.")
            
        conn = None
        cursor = None
        try:
            conn, cursor = self.get_connection_and_cursor()
            cursor.execute(query, params or ())
            if query.strip().upper().startswith("SELECT"):
                return cursor.fetchall()
            else:
                conn.commit()
                return []
        except Exception as e:
            if conn:
                conn.rollback()
            print(f"MySQL query error: {e}")
            raise
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()

_mysql_connection = None

def get_mysql_db() -> MySQLConnection:
    global _mysql_connection
    if _mysql_connection is None:
        _mysql_connection = MySQLConnection()
    return _mysql_connection

async def init_db_pool():
    await get_mysql_db().initialize_pool()