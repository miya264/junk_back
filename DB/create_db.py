#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SQLiteデータベース作成スクリプト
政策立案管理システム用のデータベースとテーブルを作成します
"""

import sqlite3
import os
from pathlib import Path

def create_database():
    """データベースを作成し、テーブルとサンプルデータを挿入"""
    
    # データベースファイルパス
    db_path = Path("policy_management.db")
    
    # 既存のデータベースファイルを削除（存在する場合）
    if db_path.exists():
        os.remove(db_path)
        print(f"既存のデータベースファイル {db_path} を削除しました")
    
    # SQLiteデータベースに接続（ファイルが存在しない場合は作成される）
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # 外部キー制約を有効化
        cursor.execute("PRAGMA foreign_keys = ON")
        
        # SQLスクリプトファイルを読み込み
        sql_file = Path("create_database.sql")
        if not sql_file.exists():
            print(f"エラー: {sql_file} が見つかりません")
            return
        
        with open(sql_file, 'r', encoding='utf-8') as f:
            sql_script = f.read()
        
        # SQLスクリプトを実行
        print("データベースとテーブルを作成中...")
        cursor.executescript(sql_script)
        
        # 変更をコミット
        conn.commit()
        
        # 作成されたテーブルを確認
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        
        print(f"\n✅ データベース作成完了!")
        print(f"作成されたテーブル数: {len(tables)}")
        print("\n作成されたテーブル:")
        for table in tables:
            print(f"  - {table[0]}")
        
        # サンプルデータの確認
        print("\n📊 サンプルデータ確認:")
        
        # 部署データ
        cursor.execute("SELECT COUNT(*) FROM departments")
        dept_count = cursor.fetchone()[0]
        print(f"  部署: {dept_count}件")
        
        # 従業員データ
        cursor.execute("SELECT COUNT(*) FROM coworkers")
        coworker_count = cursor.fetchone()[0]
        print(f"  従業員: {coworker_count}件")
        
        # プロジェクトデータ
        cursor.execute("SELECT COUNT(*) FROM projects")
        project_count = cursor.fetchone()[0]
        print(f"  プロジェクト: {project_count}件")
        
        # ステップデータ
        cursor.execute("SELECT COUNT(*) FROM policy_steps")
        step_count = cursor.fetchone()[0]
        print(f"  政策ステップ: {step_count}件")
        
        # チャットセッションデータ
        cursor.execute("SELECT COUNT(*) FROM chat_sessions")
        session_count = cursor.fetchone()[0]
        print(f"  チャットセッション: {session_count}件")
        
        # チャットメッセージデータ
        cursor.execute("SELECT COUNT(*) FROM chat_messages")
        message_count = cursor.fetchone()[0]
        print(f"  チャットメッセージ: {message_count}件")
        
        # RAG検索結果データ
        cursor.execute("SELECT COUNT(*) FROM rag_search_results")
        rag_count = cursor.fetchone()[0]
        print(f"  RAG検索結果: {rag_count}件")
        
        print(f"\n📁 データベースファイル: {db_path.absolute()}")
        print("データベースの作成が完了しました！")
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()

def verify_database():
    """データベースの内容を確認"""
    db_path = Path("policy_management.db")
    
    if not db_path.exists():
        print("データベースファイルが存在しません")
        return
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        print("\n🔍 データベース内容確認:")
        
        # テーブル一覧
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        tables = cursor.fetchall()
        
        for table in tables:
            table_name = table[0]
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cursor.fetchone()[0]
            print(f"  {table_name}: {count}件")
            
            # 最初の数件のデータを表示
            if count > 0:
                cursor.execute(f"SELECT * FROM {table_name} LIMIT 3")
                rows = cursor.fetchall()
                for i, row in enumerate(rows):
                    if i == 0:
                        print(f"    例: {row[:3]}...")  # 最初の3列のみ表示
                    break
        
    except Exception as e:
        print(f"確認中にエラーが発生しました: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    print("🚀 政策立案管理システム用データベース作成を開始します...")
    print("=" * 60)
    
    try:
        create_database()
        print("\n" + "=" * 60)
        verify_database()
        
    except Exception as e:
        print(f"\n❌ エラーが発生しました: {e}")
        exit(1)
    
    print("\n🎉 データベース作成が正常に完了しました！")
