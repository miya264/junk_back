#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
データベース接続管理
SQLiteデータベースへの接続を管理します
"""

import sqlite3
import os
from contextlib import contextmanager
from pathlib import Path


class DatabaseConnection:
    """データベース接続管理クラス"""
    
    def __init__(self, db_path: str = None):
        """
        初期化
        
        Args:
            db_path (str): データベースファイルパス。Noneの場合は環境変数から取得
        """
        if db_path is None:
            db_path = os.getenv("DATABASE_PATH", "policy_management.db")
        
        self.db_path = Path(db_path)
        
        # データベースファイルの存在確認
        if not self.db_path.exists():
            raise FileNotFoundError(f"データベースファイルが見つかりません: {self.db_path}")
    
    def get_connection(self):
        """
        データベース接続を取得
        
        Returns:
            sqlite3.Connection: SQLite接続オブジェクト
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # 辞書形式でアクセス可能にする
        conn.execute("PRAGMA foreign_keys = ON")  # 外部キー制約を有効化
        return conn
    
    @contextmanager
    def get_cursor(self):
        """
        コンテキストマネージャーでカーソルを取得
        自動でcommit/rollbackを管理
        
        Yields:
            sqlite3.Cursor: SQLiteカーソル
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            yield cursor
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
    
    @contextmanager
    def get_connection_and_cursor(self):
        """
        コンテキストマネージャーで接続とカーソルの両方を取得
        
        Yields:
            tuple: (connection, cursor)
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            yield conn, cursor
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
    
    def execute_query(self, query: str, params: tuple = None):
        """
        クエリを実行して結果を取得
        
        Args:
            query (str): SQL クエリ
            params (tuple): クエリパラメータ
            
        Returns:
            list: クエリ結果のリスト
        """
        with self.get_cursor() as cursor:
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            return cursor.fetchall()
    
    def execute_insert(self, query: str, params: tuple = None):
        """
        INSERT クエリを実行
        
        Args:
            query (str): INSERT SQL
            params (tuple): パラメータ
            
        Returns:
            int: 挿入された行のID
        """
        with self.get_cursor() as cursor:
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            return cursor.lastrowid
    
    def execute_many(self, query: str, params_list: list):
        """
        複数行の INSERT/UPDATE を実行
        
        Args:
            query (str): SQL クエリ
            params_list (list): パラメータのリスト
            
        Returns:
            int: 影響を受けた行数
        """
        with self.get_cursor() as cursor:
            cursor.executemany(query, params_list)
            return cursor.rowcount


# グローバルなデータベース接続インスタンス
_db_connection = None

def get_db():
    """
    グローバルなデータベース接続インスタンスを取得
    
    Returns:
        DatabaseConnection: データベース接続オブジェクト
    """
    global _db_connection
    if _db_connection is None:
        _db_connection = DatabaseConnection()
    return _db_connection

def init_db(db_path: str = None):
    """
    データベース接続を初期化
    
    Args:
        db_path (str): データベースファイルパス
    """
    global _db_connection
    _db_connection = DatabaseConnection(db_path)