import mysql.connector
from mysql.connector import Error
from typing import List, Dict, Any, Optional, Tuple
import os
from dotenv import load_dotenv
from pathlib import Path
from contextlib import contextmanager

# 環境変数の読み込み（backend/.env を明示的に参照）
ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=False)

class MySQLConnection:
    """MySQLデータベース接続管理クラス"""
    
    def __init__(self):
        self.connection = None
        self.config = {
            'host': os.getenv('DB_HOST'),
            'port': int(os.getenv('DB_PORT', 3306)),
            'user': os.getenv('DB_USER'),
            'password': os.getenv('DB_PASSWORD'),
            'database': os.getenv('DB_NAME'),
            'charset': 'utf8mb4',
            'collation': 'utf8mb4_unicode_ci',
            'autocommit': True,
            'pool_name': 'mypool',
            'pool_size': 5
        }
    
    def connect(self):
        """データベースに接続"""
        try:
            if self.connection is None or not self.connection.is_connected():
                self.connection = mysql.connector.connect(**self.config)
                print("MySQLデータベースに接続しました")
        except Error as e:
            print(f"MySQL接続エラー: {e}")
            raise
    
    def disconnect(self):
        """データベース接続を切断"""
        if self.connection and self.connection.is_connected():
            self.connection.close()
            print("MySQLデータベース接続を切断しました")
    
    @contextmanager
    def get_connection_and_cursor(self):
        """接続とカーソルを取得するコンテキストマネージャー"""
        connection = None
        cursor = None
        try:
            connection = mysql.connector.connect(**self.config)
            cursor = connection.cursor(dictionary=True)
            yield connection, cursor
        except Error as e:
            print(f"MySQL操作エラー: {e}")
            if connection:
                connection.rollback()
            raise
        finally:
            if cursor:
                cursor.close()
            if connection and connection.is_connected():
                connection.close()
    
    def execute_query(self, query: str, params: Optional[Tuple] = None) -> List[Dict[str, Any]]:
        """クエリを実行して結果を取得"""
        try:
            with self.get_connection_and_cursor() as (connection, cursor):
                cursor.execute(query, params or ())
                if query.strip().upper().startswith('SELECT'):
                    return cursor.fetchall()
                else:
                    connection.commit()
                    return []
        except Error as e:
            print(f"クエリ実行エラー: {e}")
            raise
    
    def execute_many(self, query: str, params_list: List[Tuple]) -> None:
        """複数のクエリを一括実行"""
        try:
            with self.get_connection_and_cursor() as (connection, cursor):
                cursor.executemany(query, params_list)
                connection.commit()
        except Error as e:
            print(f"一括実行エラー: {e}")
            raise
    
    def get_last_insert_id(self) -> int:
        """最後に挿入されたIDを取得"""
        try:
            with self.get_connection_and_cursor() as (connection, cursor):
                cursor.execute("SELECT LAST_INSERT_ID() as id")
                result = cursor.fetchone()
                return result['id'] if result else None
        except Error as e:
            print(f"ID取得エラー: {e}")
            raise

# グローバルインスタンス
_mysql_connection = None

def get_mysql_db() -> MySQLConnection:
    """MySQL接続インスタンスを取得"""
    global _mysql_connection
    if _mysql_connection is None:
        _mysql_connection = MySQLConnection()
    return _mysql_connection
