#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
データベース内容確認スクリプト
作成されたSQLiteデータベースの内容を確認します
"""

import sqlite3
from pathlib import Path

def query_database():
    """データベースの内容を確認"""
    
    db_path = Path("policy_management.db")
    
    if not db_path.exists():
        print("データベースファイルが存在しません")
        return
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        print("🔍 データベース内容詳細確認")
        print("=" * 60)
        
        # 部署情報
        print("\n📋 部署情報:")
        cursor.execute("SELECT id, name FROM departments ORDER BY id")
        departments = cursor.fetchall()
        for dept in departments:
            print(f"  {dept[0]}: {dept[1]}")
        
        # 従業員情報
        print("\n👥 従業員情報:")
        cursor.execute("""
            SELECT c.id, c.name, c.position, c.email, d.name as dept_name 
            FROM coworkers c 
            LEFT JOIN departments d ON c.department_id = d.id 
            ORDER BY c.id
        """)
        coworkers = cursor.fetchall()
        for coworker in coworkers:
            print(f"  {coworker[0]}: {coworker[1]} ({coworker[2]}) - {coworker[3]} [{coworker[4]}]")
        
        # プロジェクト情報
        print("\n📁 プロジェクト情報:")
        cursor.execute("""
            SELECT p.id, p.name, p.status, c.name as owner_name 
            FROM projects p 
            JOIN coworkers c ON p.owner_coworker_id = c.id 
            ORDER BY p.created_at
        """)
        projects = cursor.fetchall()
        for project in projects:
            print(f"  {project[0][:8]}...: {project[1]} ({project[2]}) - オーナー: {project[3]}")
        
        # 政策ステップ情報
        print("\n📊 政策ステップ情報:")
        cursor.execute("""
            SELECT ps.step_key, ps.step_name, ps.status, p.name as project_name 
            FROM policy_steps ps 
            JOIN projects p ON ps.project_id = p.id 
            ORDER BY p.name, ps.order_no
        """)
        steps = cursor.fetchall()
        for step in steps:
            print(f"  {step[0]}: {step[1]} ({step[2]}) - {step[3]}")
        
        # チャットセッション情報
        print("\n💬 チャットセッション情報:")
        cursor.execute("""
            SELECT cs.title, p.name as project_name, ps.step_name, c.name as created_by
            FROM chat_sessions cs 
            JOIN projects p ON cs.project_id = p.id 
            LEFT JOIN policy_steps ps ON cs.step_id = ps.id 
            JOIN coworkers c ON cs.created_by = c.id 
            ORDER BY cs.created_at
        """)
        sessions = cursor.fetchall()
        for session in sessions:
            step_info = f" - {session[2]}" if session[2] else ""
            print(f"  {session[0]} - {session[1]}{step_info} (作成者: {session[3]})")
        
        # チャットメッセージ情報
        print("\n💭 チャットメッセージ情報:")
        cursor.execute("""
            SELECT cm.role, cm.msg_type, SUBSTR(cm.content, 1, 50) as content_preview, 
                   c.name as sender_name, cs.title as session_title
            FROM chat_messages cm 
            JOIN chat_sessions cs ON cm.session_id = cs.id 
            JOIN coworkers c ON (cm.role = 'user' AND c.id = cs.created_by) 
            ORDER BY cm.created_at
            LIMIT 5
        """)
        messages = cursor.fetchall()
        for msg in messages:
            role_icon = "👤" if msg[0] == "user" else "🤖"
            type_info = f"[{msg[1]}]" if msg[1] != "normal" else ""
            print(f"  {role_icon} {msg[3]}: {msg[2]}... {type_info}")
        
        # RAG検索結果情報
        print("\n🔎 RAG検索結果情報:")
        cursor.execute("""
            SELECT SUBSTR(r.query, 1, 50) as query_preview, SUBSTR(r.result_text, 1, 100) as result_preview, 
                   c.name as searcher_name
            FROM rag_search_results r 
            JOIN coworkers c ON r.created_by = c.id 
            ORDER BY r.created_at
        """)
        rag_results = cursor.fetchall()
        for rag in rag_results:
            print(f"  Q: {rag[0]}...")
            print(f"     A: {rag[1]}...")
            print(f"     検索者: {rag[2]}")
        
        # テーブル構造確認
        print("\n🏗️ テーブル構造確認:")
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        tables = cursor.fetchall()
        
        for table in tables:
            table_name = table[0]
            if table_name == "sqlite_sequence":
                continue
                
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            print(f"\n  📋 {table_name}:")
            for col in columns:
                col_name, col_type, not_null, default_val, pk = col[1], col[2], col[3], col[4], col[5]
                pk_mark = " 🔑" if pk else ""
                not_null_mark = " NOT NULL" if not_null else ""
                default_mark = f" DEFAULT {default_val}" if default_val else ""
                print(f"    {col_name}: {col_type}{not_null_mark}{default_mark}{pk_mark}")
        
    except Exception as e:
        print(f"確認中にエラーが発生しました: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    query_database()
