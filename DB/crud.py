#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CRUD操作
データベースのCRUD操作を提供します
"""

from typing import List, Dict, Optional, Any
import uuid
from datetime import datetime, timezone, timedelta
from db_connection import get_db

# 日本時間（JST）のタイムゾーン設定
JST = timezone(timedelta(hours=9))

def get_jst_now():
    """現在の日本時間を取得"""
    return datetime.now(JST)


class CRUDError(Exception):
    """CRUD操作に関するエラー"""
    pass


# ========== Coworkers CRUD ==========

def search_coworkers(query: str = "", department_query: str = "") -> List[Dict[str, Any]]:
    """
    coworkersを部署名・名前で検索
    
    Args:
        query (str): 名前検索クエリ
        department_query (str): 部署名検索クエリ
        
    Returns:
        List[Dict]: 検索結果のリスト
    """
    try:
        db = get_db()
        
        sql = """
            SELECT c.id, c.name, c.position, c.email, d.name as department_name
            FROM coworkers c
            LEFT JOIN departments d ON c.department_id = d.id
            WHERE 1=1
        """
        params = []
        
        if query.strip():
            sql += " AND c.name LIKE ?"
            params.append(f"%{query.strip()}%")
            
        if department_query.strip():
            sql += " AND d.name LIKE ?"
            params.append(f"%{department_query.strip()}%")
        
        sql += " ORDER BY c.name"
        
        results = db.execute_query(sql, tuple(params))
        return [dict(row) for row in results]
        
    except Exception as e:
        raise CRUDError(f"Coworkers search failed: {str(e)}")


def get_coworker_by_id(coworker_id: int) -> Optional[Dict[str, Any]]:
    """
    IDでcoworkerを取得
    
    Args:
        coworker_id (int): coworker ID
        
    Returns:
        Optional[Dict]: coworker情報、存在しない場合はNone
    """
    try:
        db = get_db()
        
        sql = """
            SELECT c.id, c.name, c.position, c.email, d.name as department_name
            FROM coworkers c
            LEFT JOIN departments d ON c.department_id = d.id
            WHERE c.id = ?
        """
        
        results = db.execute_query(sql, (coworker_id,))
        return dict(results[0]) if results else None
        
    except Exception as e:
        raise CRUDError(f"Get coworker failed: {str(e)}")


# ========== Projects CRUD ==========

def create_project(name: str, description: str, owner_coworker_id: int, member_ids: List[int]) -> Dict[str, Any]:
    """
    プロジェクトを作成
    
    Args:
        name (str): プロジェクト名
        description (str): プロジェクト概要
        owner_coworker_id (int): オーナーのcoworker ID
        member_ids (List[int]): メンバーのcoworker IDリスト
        
    Returns:
        Dict: 作成されたプロジェクト情報
    """
    try:
        db = get_db()
        
        with db.get_connection_and_cursor() as (conn, cursor):
            # プロジェクトを作成
            project_id = str(uuid.uuid4())
            cursor.execute("""
                INSERT INTO projects (id, owner_coworker_id, name, description, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, 'active', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            """, (project_id, owner_coworker_id, name, description))
            
            # オーナーをメンバーに追加（重複回避）
            if owner_coworker_id not in member_ids:
                member_ids.append(owner_coworker_id)
            
            # プロジェクトメンバーを追加
            for coworker_id in member_ids:
                member_id = str(uuid.uuid4())
                role = 'owner' if coworker_id == owner_coworker_id else 'member'
                cursor.execute("""
                    INSERT INTO project_members (id, project_id, coworker_id, role, joined_at)
                    VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, (member_id, project_id, coworker_id, role))
            
        # 作成されたプロジェクトを取得
        return get_project_by_id(project_id)
        
    except Exception as e:
        raise CRUDError(f"Project creation failed: {str(e)}")


def get_project_by_id(project_id: str) -> Optional[Dict[str, Any]]:
    """
    プロジェクトIDで詳細取得
    
    Args:
        project_id (str): プロジェクトID
        
    Returns:
        Optional[Dict]: プロジェクト情報、存在しない場合はNone
    """
    try:
        db = get_db()
        
        # プロジェクト基本情報を取得
        project_sql = """
            SELECT p.id, p.name, p.description, p.status, p.owner_coworker_id, 
                   c.name as owner_name, p.created_at, p.updated_at
            FROM projects p
            JOIN coworkers c ON p.owner_coworker_id = c.id
            WHERE p.id = ?
        """
        
        project_results = db.execute_query(project_sql, (project_id,))
        if not project_results:
            return None
            
        project_row = project_results[0]
        
        # プロジェクトメンバーを取得
        members_sql = """
            SELECT c.id, c.name, c.position, c.email, d.name as department_name
            FROM project_members pm
            JOIN coworkers c ON pm.coworker_id = c.id
            LEFT JOIN departments d ON c.department_id = d.id
            WHERE pm.project_id = ?
            ORDER BY pm.role DESC, c.name
        """
        
        member_results = db.execute_query(members_sql, (project_id,))
        members = [dict(row) for row in member_results]
        
        return {
            'id': project_row['id'],
            'name': project_row['name'],
            'description': project_row['description'],
            'status': project_row['status'],
            'owner_coworker_id': project_row['owner_coworker_id'],
            'owner_name': project_row['owner_name'],
            'members': members,
            'created_at': project_row['created_at'],
            'updated_at': project_row['updated_at']
        }
        
    except Exception as e:
        raise CRUDError(f"Get project failed: {str(e)}")


def get_projects_by_coworker(coworker_id: int) -> List[Dict[str, Any]]:
    """
    coworkerが参加しているプロジェクトを取得
    
    Args:
        coworker_id (int): coworker ID
        
    Returns:
        List[Dict]: プロジェクトリスト（members含む）
    """
    try:
        db = get_db()
        
        # プロジェクト基本情報を取得
        sql = """
            SELECT p.id, p.name, p.description, p.status, p.owner_coworker_id,
                   c.name as owner_name, p.created_at, p.updated_at
            FROM projects p
            JOIN project_members pm ON p.id = pm.project_id
            JOIN coworkers c ON p.owner_coworker_id = c.id
            WHERE pm.coworker_id = ?
            ORDER BY p.updated_at DESC
        """
        
        results = db.execute_query(sql, (coworker_id,))
        projects = []
        
        for row in results:
            project_id = row['id']
            
            # プロジェクトメンバーを取得
            members_sql = """
                SELECT c.id, c.name, c.position, c.email, d.name as department_name
                FROM project_members pm
                JOIN coworkers c ON pm.coworker_id = c.id
                LEFT JOIN departments d ON c.department_id = d.id
                WHERE pm.project_id = ?
                ORDER BY pm.role DESC, c.name
            """
            
            member_results = db.execute_query(members_sql, (project_id,))
            members = [dict(member_row) for member_row in member_results]
            
            project = {
                'id': row['id'],
                'name': row['name'],
                'description': row['description'],
                'status': row['status'],
                'owner_coworker_id': row['owner_coworker_id'],
                'owner_name': row['owner_name'],
                'members': members,
                'created_at': row['created_at'],
                'updated_at': row['updated_at']
            }
            
            projects.append(project)
            
        return projects
        
    except Exception as e:
        raise CRUDError(f"Get projects by coworker failed: {str(e)}")


# ========== Project Step Sections CRUD ==========

def save_project_step_sections(project_id: str, step_key: str, sections: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """
    プロジェクトステップセクションをデータベースに保存
    
    Args:
        project_id (str): プロジェクトID
        step_key (str): ステップキー
        sections (List[Dict]): セクションデータ
        
    Returns:
        List[Dict]: 保存されたセクションデータ
    """
    try:
        db = get_db()
        
        # プロジェクトが存在するかチェック
        project_check = db.execute_query("SELECT id FROM projects WHERE id = ?", (project_id,))
        if not project_check:
            raise CRUDError(f"Project not found: {project_id}")
        
        print(f"DEBUG: Project {project_id} exists, proceeding with save")
        
        with db.get_connection_and_cursor() as (conn, cursor):
            # 既存データを削除
            cursor.execute("""
                DELETE FROM project_step_sections 
                WHERE project_id = ? AND step_id IN (
                    SELECT id FROM policy_steps WHERE project_id = ? AND step_key = ?
                )
            """, (project_id, project_id, step_key))
            
            # policy_stepを取得または作成
            cursor.execute("SELECT id FROM policy_steps WHERE project_id = ? AND step_key = ?", 
                         (project_id, step_key))
            step_row = cursor.fetchone()
            
            if not step_row:
                # policy_stepを作成
                step_id = str(uuid.uuid4())
                cursor.execute("""
                    INSERT INTO policy_steps (id, project_id, step_key, step_name, order_no, status, created_at, updated_at)
                    VALUES (?, ?, ?, ?, 1, 'active', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                """, (step_id, project_id, step_key, step_key.title()))
            else:
                step_id = step_row['id']
            
            # セクションを保存
            saved_sections = []
            for i, section in enumerate(sections):
                section_id = str(uuid.uuid4())
                
                # template_section_idを取得
                cursor.execute("""
                    SELECT sts.id FROM step_template_sections sts
                    JOIN step_templates st ON sts.template_id = st.id
                    WHERE st.step_key = ? AND sts.order_no = ?
                """, (step_key, i + 1))
                template_section_row = cursor.fetchone()
                template_section_id = template_section_row['id'] if template_section_row else None
                
                cursor.execute("""
                    INSERT INTO project_step_sections 
                    (id, project_id, step_id, template_section_id, order_no, section_key, label, field_type, content_text, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, 'text', ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                """, (
                    section_id, 
                    project_id, 
                    step_id, 
                    template_section_id,
                    i + 1,
                    section.get('section_key', f'section_{i+1}'),
                    section.get('label', f'Section {i+1}'),
                    section.get('content', '')
                ))
                
                saved_sections.append({
                    'id': section_id,
                    'project_id': project_id,
                    'step_key': step_key,
                    'section_key': section.get('section_key', f'section_{i+1}'),
                    'content': section.get('content', ''),
                    'created_at': get_jst_now().isoformat(),
                    'updated_at': get_jst_now().isoformat()
                })
            
            return saved_sections
        
    except Exception as e:
        raise CRUDError(f"Save project step sections failed: {str(e)}")


def get_project_step_sections(project_id: str, step_key: str) -> List[Dict[str, Any]]:
    """
    プロジェクトステップセクションを取得
    
    Args:
        project_id (str): プロジェクトID
        step_key (str): ステップキー
        
    Returns:
        List[Dict]: セクションデータ
    """
    try:
        db = get_db()
        
        sql = """
            SELECT pss.id, pss.project_id, ps.step_key, pss.section_key, pss.content_text as content, 
                   pss.created_at, pss.updated_at
            FROM project_step_sections pss
            JOIN policy_steps ps ON pss.step_id = ps.id
            WHERE pss.project_id = ? AND ps.step_key = ?
            ORDER BY pss.order_no
        """
        
        results = db.execute_query(sql, (project_id, step_key))
        return [dict(row) for row in results]
        
    except Exception as e:
        raise CRUDError(f"Get project step sections failed: {str(e)}")


# ========== Utility Functions ==========

def health_check() -> Dict[str, Any]:
    """
    データベース接続ヘルスチェック
    
    Returns:
        Dict: ヘルスチェック結果
    """
    try:
        db = get_db()
        result = db.execute_query("SELECT 1 as test")
        
        return {
            "status": "healthy",
            "database": "connected",
            "timestamp": get_jst_now().isoformat()
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "database": "connection_failed",
            "error": str(e),
            "timestamp": get_jst_now().isoformat()
        }