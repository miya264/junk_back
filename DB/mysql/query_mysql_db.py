#!/usr/bin/env python3
"""
MySQLデータベース内容確認スクリプト
"""

import os
import sys
from dotenv import load_dotenv
from mysql_connection import get_mysql_db

# 環境変数の読み込み
load_dotenv()

def query_database():
    """データベースの内容を確認"""
    
    try:
        db = get_mysql_db()
        
        print("🔍 MySQLデータベース内容確認")
        print("=" * 50)
        
        # 各テーブルの情報を表示
        tables_info = [
            ("👥 同僚情報", "coworkers", "SELECT id, name, position, email, department_name FROM coworkers ORDER BY id LIMIT 10"),
            ("📋 プロジェクト情報", "projects", "SELECT p.id, p.name, p.description, p.status, c.name as owner_name FROM projects p JOIN coworkers c ON p.owner_coworker_id = c.id ORDER BY p.created_at DESC LIMIT 10"),
            ("👥 プロジェクトメンバー情報", "project_members", "SELECT pm.project_id, p.name as project_name, c.name as member_name, pm.role FROM project_members pm JOIN projects p ON pm.project_id = p.id JOIN coworkers c ON pm.coworker_id = c.id ORDER BY pm.project_id, pm.role DESC LIMIT 15"),
            ("📝 政策ステップ情報", "policy_steps", "SELECT id, project_id, step_key, step_name, order_no, status FROM policy_steps ORDER BY project_id, order_no LIMIT 10"),
            ("📄 プロジェクトステップセクション情報", "project_step_sections", "SELECT pss.id, pss.project_id, ps.step_key, pss.section_key, LEFT(pss.content, 50) as content_preview FROM project_step_sections pss JOIN policy_steps ps ON pss.step_id = ps.id ORDER BY pss.project_id, ps.step_key, pss.section_key LIMIT 10"),
            ("💬 チャットセッション情報", "chat_sessions", "SELECT id, project_id, step_id, title, created_at FROM chat_sessions ORDER BY created_at DESC LIMIT 10"),
            ("💭 チャットメッセージ情報", "chat_messages", "SELECT cm.id, cm.session_id, cm.role, cm.msg_type, LEFT(cm.content, 50) as content_preview, cm.created_at FROM chat_messages cm ORDER BY cm.created_at DESC LIMIT 10"),
            ("🔎 RAG検索結果情報", "rag_search_results", "SELECT id, project_id, session_id, LEFT(query, 50) as query_preview, LEFT(result_text, 50) as result_preview, created_at FROM rag_search_results ORDER BY created_at DESC LIMIT 10"),
            ("📊 監査ログ情報", "audit_logs", "SELECT id, table_name, record_id, action, changed_at FROM audit_logs ORDER BY changed_at DESC LIMIT 10")
        ]
        
        for title, table_name, query in tables_info:
            print(f"\n{title}")
            print("-" * 30)
            
            try:
                results = db.execute_query(query)
                if results:
                    # ヘッダーを表示
                    headers = list(results[0].keys())
                    header_line = " | ".join(headers)
                    print(header_line)
                    print("-" * len(header_line))
                    
                    # データを表示
                    for row in results:
                        values = []
                        for header in headers:
                            value = row[header]
                            if value is None:
                                values.append("NULL")
                            elif isinstance(value, str) and len(value) > 30:
                                values.append(f"{value[:30]}...")
                            else:
                                values.append(str(value))
                        print(" | ".join(values))
                    
                    print(f"📊 表示件数: {len(results)}件")
                else:
                    print("📭 データがありません")
                    
            except Exception as e:
                print(f"❌ エラー: {e}")
        
        # 統計情報を表示
        print(f"\n📈 データベース統計")
        print("-" * 30)
        
        stats_queries = [
            ("同僚数", "SELECT COUNT(*) as count FROM coworkers"),
            ("プロジェクト数", "SELECT COUNT(*) as count FROM projects"),
            ("プロジェクトメンバー数", "SELECT COUNT(*) as count FROM project_members"),
            ("政策ステップ数", "SELECT COUNT(*) as count FROM policy_steps"),
            ("プロジェクトステップセクション数", "SELECT COUNT(*) as count FROM project_step_sections"),
            ("チャットセッション数", "SELECT COUNT(*) as count FROM chat_sessions"),
            ("チャットメッセージ数", "SELECT COUNT(*) as count FROM chat_messages"),
            ("RAG検索結果数", "SELECT COUNT(*) as count FROM rag_search_results"),
            ("監査ログ数", "SELECT COUNT(*) as count FROM audit_logs")
        ]
        
        for name, query in stats_queries:
            try:
                result = db.execute_query(query)
                count = result[0]['count'] if result else 0
                print(f"  {name}: {count}件")
            except Exception as e:
                print(f"  {name}: エラー - {e}")
        
        print(f"\n✅ データベース確認完了")
        
    except Exception as e:
        print(f"❌ データベース接続エラー: {e}")
        sys.exit(1)

if __name__ == "__main__":
    query_database()
