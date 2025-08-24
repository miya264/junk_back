# DB/mysql_crud.py  ← このファイルを丸ごと置き換え
from typing import List, Dict, Any, Optional
import uuid
from datetime import datetime
from DB.mysql_connection import get_mysql_db, MySQLConnection

class CRUDError(Exception):
    pass

def _dt_to_str(x):
    if isinstance(x, dict):
        return {k: _dt_to_str(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_dt_to_str(v) for v in x]
    if isinstance(x, datetime):
        return x.isoformat()
    return x

class MySQLCRUD:
    def __init__(self):
        self.db: MySQLConnection = get_mysql_db()

    # ---------- coworkers ----------
    def search_coworkers(self, q: str = "", department: str = "") -> List[Dict[str, Any]]:
        try:
            sql = """
                SELECT c.id, c.name, c.position, c.email, d.name AS department_name
                FROM coworkers c
                LEFT JOIN departments d ON c.department_id = d.id
                WHERE 1=1
            """
            params: list[Any] = []
            if q:
                sql += " AND (c.name LIKE %s OR c.email LIKE %s OR c.position LIKE %s)"
                t = f"%{q}%"
                params.extend([t, t, t])
            if department:
                sql += " AND d.name LIKE %s"
                params.append(f"%{department}%")
            sql += " ORDER BY c.name"
            return _dt_to_str(self.db.execute_query(sql, tuple(params) if params else None))
        except Exception as e:
            raise CRUDError(f"同僚検索エラー: {e}")

    # ---------- projects ----------
    def create_project(self, name: str, description: str, owner_coworker_id: int, member_ids: List[int] | None = None) -> Optional[Dict[str, Any]]:
        try:
            project_id = str(uuid.uuid4())
            self.db.execute_query(
                "INSERT INTO projects (id, name, description, owner_coworker_id, status) VALUES (%s,%s,%s,%s,'active')",
                (project_id, name, description, owner_coworker_id)
            )
            # owner
            self.db.execute_query(
                "INSERT INTO project_members (id, project_id, coworker_id, role) VALUES (%s,%s,%s,'owner')",
                (str(uuid.uuid4()), project_id, owner_coworker_id)
            )
            # members
            if member_ids:
                for mid in member_ids:
                    if mid == owner_coworker_id:
                        continue
                    self.db.execute_query(
                        "INSERT INTO project_members (id, project_id, coworker_id, role) VALUES (%s,%s,%s,'member')",
                        (str(uuid.uuid4()), project_id, mid)
                    )
            return self.get_project_by_id(project_id)
        except Exception as e:
            raise CRUDError(f"プロジェクト作成エラー: {e}")

    def get_project_by_id(self, project_id: str) -> Optional[Dict[str, Any]]:
        try:
            row = self.db.execute_query("""
                SELECT p.id, p.name, p.description, COALESCE(p.status,'active') AS status,
                       p.owner_coworker_id, p.created_at, p.updated_at, c.name AS owner_name
                FROM projects p
                JOIN coworkers c ON p.owner_coworker_id = c.id
                WHERE p.id = %s
            """, (project_id,))
            if not row:
                return None
            project = row[0]
            members = self.db.execute_query("""
                SELECT c.id, c.name, c.position, c.email, d.name AS department_name, pm.role
                FROM project_members pm
                JOIN coworkers c ON pm.coworker_id = c.id
                LEFT JOIN departments d ON c.department_id = d.id
                WHERE pm.project_id = %s
                ORDER BY pm.role DESC, c.name
            """, (project_id,))
            project["members"] = members
            return _dt_to_str(project)
        except Exception as e:
            raise CRUDError(f"プロジェクト取得エラー: {e}")

    def get_projects_by_coworker(self, coworker_id: int) -> List[Dict[str, Any]]:
        try:
            projects = self.db.execute_query("""
                SELECT DISTINCT p.id, p.name, p.description, COALESCE(p.status,'active') AS status,
                       p.owner_coworker_id, p.created_at, p.updated_at, c.name AS owner_name
                FROM projects p
                JOIN coworkers c ON p.owner_coworker_id = c.id
                LEFT JOIN project_members pm ON p.id = pm.project_id
                WHERE pm.coworker_id = %s OR p.owner_coworker_id = %s
                ORDER BY p.updated_at DESC
            """, (coworker_id, coworker_id))
            for p in projects:
                p["members"] = self.db.execute_query("""
                    SELECT c.id, c.name, c.position, c.email, d.name AS department_name, pm.role
                    FROM project_members pm
                    JOIN coworkers c ON pm.coworker_id = c.id
                    LEFT JOIN departments d ON c.department_id = d.id
                    WHERE pm.project_id = %s
                    ORDER BY pm.role DESC, c.name
                """, (p["id"],))
            return _dt_to_str(projects)
        except Exception as e:
            raise CRUDError(f"プロジェクト一覧取得エラー: {e}")

    # ---------- project step sections ----------
    def save_project_step_sections(self, project_id: str, step_key: str, sections: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        sections: [{ "section_key": "...", "label": "...", "content": "..." }, ...]
        """
        try:
            # project 必須
            ok = self.db.execute_query("SELECT id FROM projects WHERE id = %s", (project_id,))
            if not ok:
                raise CRUDError(f"Project not found: {project_id}")

            # step 取得/作成
            row = self.db.execute_query(
                "SELECT id FROM policy_steps WHERE project_id = %s AND step_key = %s",
                (project_id, step_key)
            )
            if row:
                step_id = row[0]["id"]
            else:
                step_id = str(uuid.uuid4())
                self.db.execute_query("""
                    INSERT INTO policy_steps (id, project_id, step_key, step_name, order_no, status, created_at, updated_at)
                    VALUES (%s,%s,%s,%s,1,'active',CURRENT_TIMESTAMP,CURRENT_TIMESTAMP)
                """, (step_id, project_id, step_key, step_key.title()))

            # 既存削除（同 step_key）
            self.db.execute_query("""
                DELETE pss FROM project_step_sections pss
                JOIN policy_steps ps ON pss.step_id = ps.id
                WHERE pss.project_id = %s AND ps.project_id = %s AND ps.step_key = %s
            """, (project_id, project_id, step_key))

            saved: List[Dict[str, Any]] = []
            for i, sec in enumerate(sections, start=1):
                # テンプレ紐づけ（あれば）
                t = self.db.execute_query("""
                    SELECT sts.id
                    FROM step_template_sections sts
                    JOIN step_templates st ON sts.template_id = st.id
                    WHERE st.step_key = %s AND sts.order_no = %s
                """, (step_key, i))
                template_section_id = t[0]["id"] if t else None

                section_id = str(uuid.uuid4())
                self.db.execute_query("""
                    INSERT INTO project_step_sections
                      (id, project_id, step_id, template_section_id, order_no,
                       section_key, label, field_type, content_text, created_at, updated_at)
                    VALUES
                      (%s,%s,%s,%s,%s,%s,%s,'text',%s,CURRENT_TIMESTAMP,CURRENT_TIMESTAMP)
                """, (
                    section_id, project_id, step_id, template_section_id, i,
                    sec.get("section_key", f"section_{i}"),
                    sec.get("label", f"Section {i}"),
                    sec.get("content", "")
                ))

                now = datetime.now().isoformat()
                saved.append({
                    "id": section_id,
                    "project_id": project_id,
                    "step_key": step_key,
                    "section_key": sec.get("section_key", f"section_{i}"),
                    "content": sec.get("content", ""),
                    "created_at": now,
                    "updated_at": now,
                })
            return saved
        except Exception as e:
            raise CRUDError(f"ステップセクション保存エラー: {e}")

    def get_project_step_sections(self, project_id: str, step_key: str) -> List[Dict[str, Any]]:
        try:
            rows = self.db.execute_query("""
                SELECT pss.id,
                       pss.project_id,
                       ps.step_key,
                       pss.section_key,
                       pss.content_text AS content,   -- ← ここが重要
                       pss.created_at,
                       pss.updated_at
                FROM project_step_sections pss
                JOIN policy_steps ps ON pss.step_id = ps.id
                WHERE pss.project_id = %s AND ps.step_key = %s
                ORDER BY pss.order_no
            """, (project_id, step_key))
            return _dt_to_str(rows)
        except Exception as e:
            raise CRUDError(f"ステップセクション取得エラー: {e}")

    # ---------- health ----------
    def health_check(self) -> Dict[str, Any]:
        try:
            tables = [
                'coworkers','departments','auth_users',
                'projects','project_members','policy_steps',
                'step_templates','step_template_sections','project_step_sections',
                'chat_sessions','chat_messages','rag_search_results','rag_result_sources','audit_logs'
            ]
            counts = {}
            for t in tables:
                r = self.db.execute_query(f"SELECT COUNT(*) AS cnt FROM {t}")
                counts[t] = r[0]['cnt'] if r else 0
            return {'status':'healthy','database':'MySQL','table_counts':counts}
        except Exception as e:
            return {'status':'unhealthy','database':'MySQL','error':str(e)}
        
    def fetch_project_step_sections(self, project_id: str, step_key: str) -> List[Dict[str, Any]]:
        """
        policy_steps.step_key をキーに、そのステップ配下の project_step_sections を
        ラベル付きで取得（エージェントの SectionRepo 想定スキーマに一致）
        返却キー: id, section_key, label, content_text, updated_at
        """
        try:
            rows = self.db.execute_query("""
                SELECT
                    pss.id,
                    pss.section_key,
                    pss.label,
                    pss.content_text,
                    pss.updated_at
                FROM policy_steps ps
                JOIN project_step_sections pss ON pss.step_id = ps.id
                WHERE ps.project_id = %s
                  AND ps.step_key   = %s
                ORDER BY pss.order_no
            """, (project_id, step_key))
            # フィールド名を SectionRepo 仕様に合わせる
            data = []
            for r in rows:
                data.append({
                    "id": r["id"],
                    "section_key": r["section_key"],
                    "label": r.get("label"),
                    "content_text": r.get("content_text") or "",
                    "updated_at": r.get("updated_at"),
                })
            return _dt_to_str(data)
        except Exception as e:
            raise CRUDError(f"ステップセクション(ラベル付き)取得エラー: {e}")
        
    def get_recent_chat_messages(self, session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        指定セッションの直近メッセージを最大 limit 件だけ取得。
        返却は **古い→新しい（昇順）** に並べ替えて返すので、そのまま文脈に使えます。
        """
        try:
            rows = self.db.execute_query("""
                SELECT role, msg_type, content, created_at
                FROM (
                    SELECT role, msg_type, content, created_at
                    FROM chat_messages
                    WHERE session_id = %s
                    ORDER BY created_at DESC
                    LIMIT %s
                ) t
                ORDER BY created_at ASC
            """, (session_id, limit))
            return _dt_to_str(rows)
        except Exception as e:
            raise CRUDError(f"チャット履歴取得エラー: {e}")


# global instance & aliases
mysql_crud = MySQLCRUD()

def search_coworkers(q: str = "", department: str = "") -> List[Dict[str, Any]]:
    return mysql_crud.search_coworkers(q, department)

def create_project(name: str, description: str, owner_coworker_id: int, member_ids: List[int] | None = None) -> Optional[Dict[str, Any]]:
    return mysql_crud.create_project(name, description, owner_coworker_id, member_ids)

def get_project_by_id(project_id: str) -> Optional[Dict[str, Any]]:
    return mysql_crud.get_project_by_id(project_id)

def get_projects_by_coworker(coworker_id: int) -> List[Dict[str, Any]]:
    return mysql_crud.get_projects_by_coworker(coworker_id)

def save_project_step_sections(project_id: str, step_key: str, sections: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    return mysql_crud.save_project_step_sections(project_id, step_key, sections)

def get_project_step_sections(project_id: str, step_key: str) -> List[Dict[str, Any]]:
    return mysql_crud.get_project_step_sections(project_id, step_key)

def get_recent_chat_messages(session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
    return mysql_crud.get_recent_chat_messages(session_id, limit)


def health_check() -> Dict[str, Any]:
    return mysql_crud.health_check()
