-- パフォーマンス向上用インデックス追加スクリプト
-- 実行前に必ずバックアップを取得してください

-- 1. coworkersテーブルの検索用複合インデックス（最も頻繁に使用）
CREATE INDEX idx_coworkers_search ON coworkers (name, email, position);
CREATE INDEX idx_coworkers_department ON coworkers (department_id, name);

-- 2. project_step_sectionsの複合インデックス（最も重要）
CREATE INDEX idx_pss_project_step ON project_step_sections (project_id, step_id);
CREATE INDEX idx_pss_project_order ON project_step_sections (project_id, order_no);

-- 3. project_membersの複合インデックス
CREATE INDEX idx_pm_coworker_project ON project_members (coworker_id, project_id);
CREATE INDEX idx_pm_project_role ON project_members (project_id, role);

-- 4. chat_messagesのパフォーマンス改善
CREATE INDEX idx_chat_session_timestamp ON chat_messages (session_id, created_at DESC);

-- 5. policy_stepsのstep_key検索用（サブクエリ最適化）
CREATE INDEX idx_policy_steps_step_key ON policy_steps (step_key);

-- 6. projectsテーブルの最適化
CREATE INDEX idx_projects_owner_updated ON projects (owner_coworker_id, updated_at DESC);
CREATE INDEX idx_projects_status_updated ON projects (status, updated_at DESC);

-- 7. rag_search_resultsの最適化
CREATE INDEX idx_rag_project_step_session ON rag_search_results (project_id, step_id, session_id);

-- 8. departmentsの名前検索用
CREATE INDEX idx_departments_name ON departments (name);

-- インデックスの確認
SHOW INDEX FROM coworkers;
SHOW INDEX FROM project_step_sections;
SHOW INDEX FROM project_members;
SHOW INDEX FROM chat_messages;
SHOW INDEX FROM policy_steps;
SHOW INDEX FROM projects;
SHOW INDEX FROM departments;

-- パフォーマンス確認クエリ
-- EXPLAIN SELECT c.id, c.name, c.position, c.email, d.name AS department_name FROM coworkers c LEFT JOIN departments d ON c.department_id = d.id WHERE c.name LIKE '%test%' ORDER BY c.name LIMIT 100;