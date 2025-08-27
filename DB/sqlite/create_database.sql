-- SQLite Database Creation Script for Policy Management System
PRAGMA foreign_keys = ON;

-- 認証・ユーザー管理
CREATE TABLE IF NOT EXISTS auth_users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    coworker_id INTEGER NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    last_login TIMESTAMP,
    FOREIGN KEY (coworker_id) REFERENCES coworkers(id)
);

CREATE TABLE IF NOT EXISTS coworkers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name VARCHAR(255) NOT NULL,
    position VARCHAR(255),
    email VARCHAR(255) NOT NULL UNIQUE,
    sso_id VARCHAR(255) NOT NULL UNIQUE,
    department_id INTEGER,
    FOREIGN KEY (department_id) REFERENCES departments(id)
);

CREATE TABLE IF NOT EXISTS departments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name VARCHAR(255) NOT NULL UNIQUE
);

-- プロジェクト管理
CREATE TABLE IF NOT EXISTS projects (
    id TEXT PRIMARY KEY,  -- UUID
    owner_coworker_id INTEGER NOT NULL,
    name VARCHAR(200) NOT NULL,
    description TEXT,
    status VARCHAR(40), -- active, archived など
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (owner_coworker_id) REFERENCES coworkers(id)
);

CREATE TABLE IF NOT EXISTS project_members (
    id TEXT PRIMARY KEY,  -- UUID
    project_id TEXT NOT NULL,
    coworker_id INTEGER NOT NULL,
    role VARCHAR(40),
    joined_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (project_id) REFERENCES projects(id),
    FOREIGN KEY (coworker_id) REFERENCES coworkers(id),
    UNIQUE(project_id, coworker_id)
);

-- 可変ステップ（プロジェクト内の実体）
CREATE TABLE IF NOT EXISTS policy_steps (
    id TEXT PRIMARY KEY,  -- UUID
    project_id TEXT NOT NULL,
    step_key VARCHAR(60) NOT NULL,
    step_name VARCHAR(120) NOT NULL,
    order_no INTEGER NOT NULL,
    status VARCHAR(40),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (project_id) REFERENCES projects(id),
    UNIQUE(project_id, step_key)
);

-- ステップテンプレート（定義側）
CREATE TABLE IF NOT EXISTS step_templates (
    id TEXT PRIMARY KEY,  -- UUID
    step_key VARCHAR(60) NOT NULL,
    title TEXT NOT NULL,
    version INTEGER DEFAULT 1,
    is_active BOOLEAN DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(step_key, version)
);

CREATE TABLE IF NOT EXISTS step_template_sections (
    id TEXT PRIMARY KEY,  -- UUID
    template_id TEXT NOT NULL,
    order_no INTEGER NOT NULL,
    section_key VARCHAR(50),
    label TEXT NOT NULL,
    field_type VARCHAR(30) NOT NULL DEFAULT 'text',
    options_json TEXT,
    validation_schema TEXT,
    default_placeholder TEXT,
    default_help TEXT,
    is_required BOOLEAN DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (template_id) REFERENCES step_templates(id),
    UNIQUE(template_id, order_no)
);

-- プロジェクトステップセクション（実体）
CREATE TABLE IF NOT EXISTS project_step_sections (
    id TEXT PRIMARY KEY,  -- UUID
    project_id TEXT NOT NULL,
    step_id TEXT NOT NULL,
    template_section_id TEXT,
    order_no INTEGER NOT NULL,
    section_key VARCHAR(50),
    label TEXT NOT NULL,
    field_type VARCHAR(30) NOT NULL DEFAULT 'text',
    content_text TEXT,
    content_json TEXT,
    is_required BOOLEAN,
    created_by INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (project_id) REFERENCES projects(id),
    FOREIGN KEY (step_id) REFERENCES policy_steps(id),
    FOREIGN KEY (template_section_id) REFERENCES step_template_sections(id),
    FOREIGN KEY (created_by) REFERENCES coworkers(id),
    UNIQUE(step_id, order_no)
);

-- チャット
CREATE TABLE IF NOT EXISTS chat_sessions (
    id TEXT PRIMARY KEY,  -- UUID
    project_id TEXT NOT NULL,
    step_id TEXT,
    title VARCHAR(200),
    created_by INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (project_id) REFERENCES projects(id),
    FOREIGN KEY (step_id) REFERENCES policy_steps(id),
    FOREIGN KEY (created_by) REFERENCES coworkers(id)
);

CREATE TABLE IF NOT EXISTS chat_messages (
    id TEXT PRIMARY KEY,  -- UUID
    session_id TEXT NOT NULL,
    project_id TEXT NOT NULL,
    step_id TEXT,
    role VARCHAR(20) NOT NULL,
    msg_type VARCHAR(20) NOT NULL,
    content TEXT NOT NULL,
    content_json TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES chat_sessions(id),
    FOREIGN KEY (project_id) REFERENCES projects(id),
    FOREIGN KEY (step_id) REFERENCES policy_steps(id)
);

-- RAG
CREATE TABLE IF NOT EXISTS rag_search_results (
    id TEXT PRIMARY KEY,  -- UUID
    project_id TEXT NOT NULL,
    step_id TEXT NOT NULL,
    session_id TEXT,
    query TEXT NOT NULL,
    result_text TEXT,
    result_json TEXT,
    sources_json TEXT,
    created_by INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (project_id) REFERENCES projects(id),
    FOREIGN KEY (step_id) REFERENCES policy_steps(id),
    FOREIGN KEY (session_id) REFERENCES chat_sessions(id),
    FOREIGN KEY (created_by) REFERENCES coworkers(id)
);

CREATE TABLE IF NOT EXISTS rag_result_sources (
    id TEXT PRIMARY KEY,  -- UUID
    rag_result_id TEXT NOT NULL,
    source_title VARCHAR(300),
    source_url VARCHAR(500),
    section_title VARCHAR(300),
    page_number INTEGER,
    year_label VARCHAR(20),
    score REAL,
    FOREIGN KEY (rag_result_id) REFERENCES rag_search_results(id)
);

-- 監査ログ
CREATE TABLE IF NOT EXISTS audit_logs (
    id TEXT PRIMARY KEY,  -- UUID
    project_id TEXT,
    step_id TEXT,
    session_id TEXT,
    coworker_id INTEGER,
    action VARCHAR(50),
    detail TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (project_id) REFERENCES projects(id),
    FOREIGN KEY (step_id) REFERENCES policy_steps(id),
    FOREIGN KEY (session_id) REFERENCES chat_sessions(id),
    FOREIGN KEY (coworker_id) REFERENCES coworkers(id)
);

-- インデックス
CREATE INDEX IF NOT EXISTS idx_projects_owner ON projects(owner_coworker_id);
CREATE INDEX IF NOT EXISTS idx_project_members_project ON project_members(project_id);
CREATE INDEX IF NOT EXISTS idx_project_members_coworker ON project_members(coworker_id);
CREATE INDEX IF NOT EXISTS idx_policy_steps_project ON policy_steps(project_id);
CREATE INDEX IF NOT EXISTS idx_step_template_sections_template ON step_template_sections(template_id);
CREATE INDEX IF NOT EXISTS idx_project_step_sections_project ON project_step_sections(project_id);
CREATE INDEX IF NOT EXISTS idx_project_step_sections_step ON project_step_sections(step_id);
CREATE INDEX IF NOT EXISTS idx_chat_sessions_project ON chat_sessions(project_id);
CREATE INDEX IF NOT EXISTS idx_chat_sessions_step ON chat_sessions(step_id);
CREATE INDEX IF NOT EXISTS idx_chat_messages_session ON chat_messages(session_id);
CREATE INDEX IF NOT EXISTS idx_chat_messages_project_step ON chat_messages(project_id, step_id);
CREATE INDEX IF NOT EXISTS idx_chat_messages_type ON chat_messages(msg_type);
CREATE INDEX IF NOT EXISTS idx_rag_search_results_project_step ON rag_search_results(project_id, step_id);
CREATE INDEX IF NOT EXISTS idx_rag_search_results_session ON rag_search_results(session_id);
CREATE INDEX IF NOT EXISTS idx_rag_search_results_created ON rag_search_results(created_at);
CREATE INDEX IF NOT EXISTS idx_rag_result_sources_result ON rag_result_sources(rag_result_id);
CREATE INDEX IF NOT EXISTS idx_audit_logs_project ON audit_logs(project_id);
CREATE INDEX IF NOT EXISTS idx_audit_logs_step ON audit_logs(step_id);
CREATE INDEX IF NOT EXISTS idx_audit_logs_session ON audit_logs(session_id);
CREATE INDEX IF NOT EXISTS idx_audit_logs_coworker ON audit_logs(coworker_id);
CREATE INDEX IF NOT EXISTS idx_audit_logs_action ON audit_logs(action);
CREATE INDEX IF NOT EXISTS idx_audit_logs_created ON audit_logs(created_at);

BEGIN;

-- マスタ（単行INSERT）
INSERT OR IGNORE INTO departments (id, name) VALUES (1, '政策企画部');
INSERT OR IGNORE INTO departments (id, name) VALUES (2, 'IT推進部');
INSERT OR IGNORE INTO departments (id, name) VALUES (3, '総務部');

INSERT OR IGNORE INTO coworkers (id, name, position, email, sso_id, department_id) VALUES (1, '田中 太郎', '課長', 'tanaka@example.com', 'tanaka001', 1);
INSERT OR IGNORE INTO coworkers (id, name, position, email, sso_id, department_id) VALUES (2, '佐藤 花子', '主任', 'sato@example.com', 'sato002', 1);
INSERT OR IGNORE INTO coworkers (id, name, position, email, sso_id, department_id) VALUES (3, '鈴木 次郎', '係長', 'suzuki@example.com', 'suzuki003', 2);

INSERT OR IGNORE INTO auth_users (id, coworker_id, password_hash, last_login) VALUES (1001, 1, 'hashed_password_1', NULL);
INSERT OR IGNORE INTO auth_users (id, coworker_id, password_hash, last_login) VALUES (1002, 2, 'hashed_password_2', NULL);
INSERT OR IGNORE INTO auth_users (id, coworker_id, password_hash, last_login) VALUES (1003, 3, 'hashed_password_3', NULL);

-- プロジェクト／ステップ（単行INSERT）
INSERT OR IGNORE INTO projects (id, owner_coworker_id, name, description, status) VALUES ('550e8400-e29b-41d4-a716-446655440000', 1, '地域活性化プロジェクト', '地域経済の活性化を目指す政策立案プロジェクト', 'active');
INSERT OR IGNORE INTO projects (id, owner_coworker_id, name, description, status) VALUES ('550e8400-e29b-41d4-a716-446655440001', 2, 'デジタル化推進', '行政サービスのデジタル化推進プロジェクト', 'active');

INSERT OR IGNORE INTO policy_steps (id, project_id, step_key, step_name, order_no, status) VALUES ('550e8400-e29b-41d4-a716-446655440010', '550e8400-e29b-41d4-a716-446655440000', 'analysis',  '現状分析・課題整理', 1, 'in_progress');
INSERT OR IGNORE INTO policy_steps (id, project_id, step_key, step_name, order_no, status) VALUES ('550e8400-e29b-41d4-a716-446655440011', '550e8400-e29b-41d4-a716-446655440000', 'objective', '目的整理',           2, 'draft');
INSERT OR IGNORE INTO policy_steps (id, project_id, step_key, step_name, order_no, status) VALUES ('550e8400-e29b-41d4-a716-446655440012', '550e8400-e29b-41d4-a716-446655440000', 'concept',   'コンセプト策定',     3, 'draft');
INSERT OR IGNORE INTO policy_steps (id, project_id, step_key, step_name, order_no, status) VALUES ('550e8400-e29b-41d4-a716-446655440013', '550e8400-e29b-41d4-a716-446655440000', 'plan',      '施策案作成',         4, 'draft');
INSERT OR IGNORE INTO policy_steps (id, project_id, step_key, step_name, order_no, status) VALUES ('550e8400-e29b-41d4-a716-446655440014', '550e8400-e29b-41d4-a716-446655440000', 'proposal',  '提案書作成',         5, 'draft');

INSERT OR IGNORE INTO step_templates (id, step_key, title, version, is_active) VALUES ('550e8400-e29b-41d4-a716-446655440020', 'analysis',  '現状分析・課題整理', 1, 1);
INSERT OR IGNORE INTO step_templates (id, step_key, title, version, is_active) VALUES ('550e8400-e29b-41d4-a716-446655440021', 'objective', '目的整理',           1, 1);
INSERT OR IGNORE INTO step_templates (id, step_key, title, version, is_active) VALUES ('550e8400-e29b-41d4-a716-446655440022', 'concept',   'コンセプト策定',     1, 1);
INSERT OR IGNORE INTO step_templates (id, step_key, title, version, is_active) VALUES ('550e8400-e29b-41d4-a716-446655440023', 'plan',      '施策案作成',         1, 1);
INSERT OR IGNORE INTO step_templates (id, step_key, title, version, is_active) VALUES ('550e8400-e29b-41d4-a716-446655440024', 'proposal',  '提案書作成',         1, 1);

-- step_template_sections（UPSERT：1行ずつ）
-- ANALYSIS
INSERT INTO step_template_sections (id, template_id, order_no, section_key, label, field_type, default_placeholder, default_help, is_required, created_at, updated_at)
VALUES ('550e8400-e29b-41d4-a716-446655440030', '550e8400-e29b-41d4-a716-446655440020', 1, 'problem',    '課題と裏付け',    'richtext', '統計/調査/事例の根拠', '課題と裏付け（定量・定性）を記入してください', 1, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
ON CONFLICT(template_id, order_no) DO UPDATE SET section_key=excluded.section_key, label=excluded.label, field_type=excluded.field_type, default_placeholder=excluded.default_placeholder, default_help=excluded.default_help, is_required=excluded.is_required, updated_at=CURRENT_TIMESTAMP;

INSERT INTO step_template_sections (id, template_id, order_no, section_key, label, field_type, default_placeholder, default_help, is_required, created_at, updated_at)
VALUES ('550e8400-e29b-41d4-a716-446655440031', '550e8400-e29b-41d4-a716-446655440020', 2, 'background', '背景構造の評価',  'text',     '制度・市場・競合…',   '課題の背景にある構造（制度・市場など）を簡単に評価してください', 1, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
ON CONFLICT(template_id, order_no) DO UPDATE SET section_key=excluded.section_key, label=excluded.label, field_type=excluded.field_type, default_placeholder=excluded.default_placeholder, default_help=excluded.default_help, is_required=excluded.is_required, updated_at=CURRENT_TIMESTAMP;

INSERT INTO step_template_sections (id, template_id, order_no, section_key, label, field_type, default_placeholder, default_help, is_required, created_at, updated_at)
VALUES ('550e8400-e29b-41d4-a716-446655440032', '550e8400-e29b-41d4-a716-446655440020', 3, 'priority',   '優先度と理由',    'text',     '優先度と根拠',        '解決すべき課題の優先度と理由を整理しましょう', 1, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
ON CONFLICT(template_id, order_no) DO UPDATE SET section_key=excluded.section_key, label=excluded.label, field_type=excluded.field_type, default_placeholder=excluded.default_placeholder, default_help=excluded.default_help, is_required=excluded.is_required, updated_at=CURRENT_TIMESTAMP;

-- OBJECTIVE
INSERT INTO step_template_sections (id, template_id, order_no, section_key, label, field_type, default_placeholder, default_help, is_required, created_at, updated_at)
VALUES ('550e8400-e29b-41d4-a716-446655440040', '550e8400-e29b-41d4-a716-446655440021', 1, 'goal',        '最終ゴール',     'text', '具体的な到達像',      '最終的に達成したいゴールを具体的に記載してください', 1, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
ON CONFLICT(template_id, order_no) DO UPDATE SET section_key=excluded.section_key, label=excluded.label, field_type=excluded.field_type, default_placeholder=excluded.default_placeholder, default_help=excluded.default_help, is_required=excluded.is_required, updated_at=CURRENT_TIMESTAMP;

INSERT INTO step_template_sections (id, template_id, order_no, section_key, label, field_type, default_placeholder, default_help, is_required, created_at, updated_at)
VALUES ('550e8400-e29b-41d4-a716-446655440041', '550e8400-e29b-41d4-a716-446655440021', 2, 'kpi',         'KPI・目標値',    'text', 'いつまでに/どれだけ', 'KPI・目標値（いつまでに・どれだけ）を記入してください', 1, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
ON CONFLICT(template_id, order_no) DO UPDATE SET section_key=excluded.section_key, label=excluded.label, field_type=excluded.field_type, default_placeholder=excluded.default_placeholder, default_help=excluded.default_help, is_required=excluded.is_required, updated_at=CURRENT_TIMESTAMP;

INSERT INTO step_template_sections (id, template_id, order_no, section_key, label, field_type, default_placeholder, default_help, is_required, created_at, updated_at)
VALUES ('550e8400-e29b-41d4-a716-446655440042', '550e8400-e29b-41d4-a716-446655440021', 3, 'constraints', '前提条件・制約', 'text', '予算/人員/期間',      '前提条件・制約（予算、人員、期間など）を整理しましょう', 0, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
ON CONFLICT(template_id, order_no) DO UPDATE SET section_key=excluded.section_key, label=excluded.label, field_type=excluded.field_type, default_placeholder=excluded.default_placeholder, default_help=excluded.default_help, is_required=excluded.is_required, updated_at=CURRENT_TIMESTAMP;

-- CONCEPT
INSERT INTO step_template_sections (id, template_id, order_no, section_key, label, field_type, default_placeholder, default_help, is_required, created_at, updated_at)
VALUES ('550e8400-e29b-41d4-a716-446655440050', '550e8400-e29b-41d4-a716-446655440022', 1, 'policy',     '基本方針',       'richtext', '価値/対象/提供方法', '基本方針（どんな価値を誰に、どう届けるか）を記入してください', 1, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
ON CONFLICT(template_id, order_no) DO UPDATE SET section_key=excluded.section_key, label=excluded.label, field_type=excluded.field_type, default_placeholder=excluded.default_placeholder, default_help=excluded.default_help, is_required=excluded.is_required, updated_at=CURRENT_TIMESTAMP;

INSERT INTO step_template_sections (id, template_id, order_no, section_key, label, field_type, default_placeholder, default_help, is_required, created_at, updated_at)
VALUES ('550e8400-e29b-41d4-a716-446655440051', '550e8400-e29b-41d4-a716-446655440022', 2, 'rationale',  '根拠・示唆',     'text',     '調査/事例/専門家意見','方針の根拠・示唆（調査、事例、専門家意見など）を書いてください', 0, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
ON CONFLICT(template_id, order_no) DO UPDATE SET section_key=excluded.section_key, label=excluded.label, field_type=excluded.field_type, default_placeholder=excluded.default_placeholder, default_help=excluded.default_help, is_required=excluded.is_required, updated_at=CURRENT_TIMESTAMP;

INSERT INTO step_template_sections (id, template_id, order_no, section_key, label, field_type, default_placeholder, default_help, is_required, created_at, updated_at)
VALUES ('550e8400-e29b-41d4-a716-446655440052', '550e8400-e29b-41d4-a716-446655440022', 3, 'risks',      'リスクと打ち手', 'text',     '代替案/実験設計',     '主要リスクと打ち手（代替案、実験設計）を整理しましょう', 0, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
ON CONFLICT(template_id, order_no) DO UPDATE SET section_key=excluded.section_key, label=excluded.label, field_type=excluded.field_type, default_placeholder=excluded.default_placeholder, default_help=excluded.default_help, is_required=excluded.is_required, updated_at=CURRENT_TIMESTAMP;

-- PLAN
INSERT INTO step_template_sections (id, template_id, order_no, section_key, label, field_type, default_placeholder, default_help, is_required, created_at, updated_at)
VALUES ('550e8400-e29b-41d4-a716-446655440060', '550e8400-e29b-41d4-a716-446655440023', 1, 'initiatives', '主な施策',           'text', '3〜5施策の概要と狙い', '主な施策（3〜5個）の概要と狙いを整理してください', 1, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
ON CONFLICT(template_id, order_no) DO UPDATE SET section_key=excluded.section_key, label=excluded.label, field_type=excluded.field_type, default_placeholder=excluded.default_placeholder, default_help=excluded.default_help, is_required=excluded.is_required, updated_at=CURRENT_TIMESTAMP;

INSERT INTO step_template_sections (id, template_id, order_no, section_key, label, field_type, default_placeholder, default_help, is_required, created_at, updated_at)
VALUES ('550e8400-e29b-41d4-a716-446655440061', '550e8400-e29b-41d4-a716-446655440023', 2, 'schedule',    '体制・スケジュール', 'text', '役割分担/マイルストーン','体制・役割分担・スケジュールを記入してください', 0, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
ON CONFLICT(template_id, order_no) DO UPDATE SET section_key=excluded.section_key, label=excluded.label, field_type=excluded.field_type, default_placeholder=excluded.default_placeholder, default_help=excluded.default_help, is_required=excluded.is_required, updated_at=CURRENT_TIMESTAMP;

INSERT INTO step_template_sections (id, template_id, order_no, section_key, label, field_type, default_placeholder, default_help, is_required, created_at, updated_at)
VALUES ('550e8400-e29b-41d4-a716-446655440062', '550e8400-e29b-41d4-a716-446655440023', 3, 'cost',        'コストと効果',       'text', '概算コスト/効果/根拠',  '概算コスト・効果見込み（根拠も）を書いてください', 0, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
ON CONFLICT(template_id, order_no) DO UPDATE SET section_key=excluded.section_key, label=excluded.label, field_type=excluded.field_type, default_placeholder=excluded.default_placeholder, default_help=excluded.default_help, is_required=excluded.is_required, updated_at=CURRENT_TIMESTAMP;

-- PROPOSAL
INSERT INTO step_template_sections (id, template_id, order_no, section_key, label, field_type, default_placeholder, default_help, is_required, created_at, updated_at)
VALUES ('550e8400-e29b-41d4-a716-446655440070', '550e8400-e29b-41d4-a716-446655440024', 1, 'summary',    '提案サマリー',       'richtext', '背景→課題→解決→効果→体制', '提案のサマリー（背景→課題→解決→効果→体制）を書いてください', 1, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
ON CONFLICT(template_id, order_no) DO UPDATE SET section_key=excluded.section_key, label=excluded.label, field_type=excluded.field_type, default_placeholder=excluded.default_placeholder, default_help=excluded.default_help, is_required=excluded.is_required, updated_at=CURRENT_TIMESTAMP;

INSERT INTO step_template_sections (id, template_id, order_no, section_key, label, field_type, default_placeholder, default_help, is_required, created_at, updated_at)
VALUES ('550e8400-e29b-41d4-a716-446655440071', '550e8400-e29b-41d4-a716-446655440024', 2, 'concerns',   '意思決定者の関心',   'text',     '費用対効果/リスク/責任分担', '意思決定者の関心（費用対効果、リスク、責任分担）を整理してください', 0, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
ON CONFLICT(template_id, order_no) DO UPDATE SET section_key=excluded.section_key, label=excluded.label, field_type=excluded.field_type, default_placeholder=excluded.default_placeholder, default_help=excluded.default_help, is_required=excluded.is_required, updated_at=CURRENT_TIMESTAMP;

INSERT INTO step_template_sections (id, template_id, order_no, section_key, label, field_type, default_placeholder, default_help, is_required, created_at, updated_at)
VALUES ('550e8400-e29b-41d4-a716-446655440072', '550e8400-e29b-41d4-a716-446655440024', 3, 'next_steps', '次のアクション',     'text',     '承認/説明/PoC準備',         '次のアクション（承認プロセス、関係者説明、PoC準備など）を記入してください', 0, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
ON CONFLICT(template_id, order_no) DO UPDATE SET section_key=excluded.section_key, label=excluded.label, field_type=excluded.field_type, default_placeholder=excluded.default_placeholder, default_help=excluded.default_help, is_required=excluded.is_required, updated_at=CURRENT_TIMESTAMP;

-- 既存 project_step_sections の整備（UPDATEはそのまま）
UPDATE project_step_sections SET section_key='problem',    label='課題と裏付け',     field_type='richtext' WHERE id='550e8400-e29b-41d4-a716-446655440040';
UPDATE project_step_sections SET section_key='background', label='背景構造の評価',   field_type='text'     WHERE id='550e8400-e29b-41d4-a716-446655440041';
UPDATE project_step_sections SET section_key='priority',   label='優先度と理由',     field_type='text'     WHERE id='550e8400-e29b-41d4-a716-446655440042';

-- チャット / RAG / 監査（単行INSERT）
INSERT OR IGNORE INTO chat_sessions (id, project_id, step_id, title, created_by) VALUES ('550e8400-e29b-41d4-a716-446655440050', '550e8400-e29b-41d4-a716-446655440000', '550e8400-e29b-41d4-a716-446655440010', '現状分析についての相談', 1);

INSERT OR IGNORE INTO chat_messages (id, session_id, project_id, step_id, role, msg_type, content, created_at) VALUES ('550e8400-e29b-41d4-a716-446655440060', '550e8400-e29b-41d4-a716-446655440050', '550e8400-e29b-41d4-a716-446655440000', '550e8400-e29b-41d4-a716-446655440010', 'user', 'normal', '地域活性化の現状分析を進めたいのですが、どのような観点で整理すればよいでしょうか？', '2024-01-15 10:00:00');
INSERT OR IGNORE INTO chat_messages (id, session_id, project_id, step_id, role, msg_type, content, created_at) VALUES ('550e8400-e29b-41d4-a716-446655440061', '550e8400-e29b-41d4-a716-446655440050', '550e8400-e29b-41d4-a716-446655440000', '550e8400-e29b-41d4-a716-446655440010', 'ai', 'normal', '地域活性化の現状分析では、以下の3つの観点で整理することをお勧めします：\n\n1. **課題と裏付け（定量・定性）**: 具体的なデータや事実に基づく課題の特定\n2. **課題の背景にある構造**: 制度や市場の構造的な問題の分析\n3. **解決すべき課題の優先度**: 影響度と緊急度を考慮した優先順位付け\n\n既にプロジェクトで設定されているセクションに沿って整理していきましょう。', '2024-01-15 10:01:00');

INSERT OR IGNORE INTO rag_search_results (id, project_id, step_id, session_id, query, result_text, sources_json, created_by) VALUES ('550e8400-e29b-41d4-a716-446655440070', '550e8400-e29b-41d4-a716-446655440000', '550e8400-e29b-41d4-a716-446655440010', '550e8400-e29b-41d4-a716-446655440050', '地域活性化 成功事例', '地域活性化の成功事例として、以下のような取り組みが報告されています：\n\n1. 地域資源を活用した観光振興\n2. 地元企業との連携による産業振興\n3. 住民参加型のまちづくり\n\nこれらの事例を参考に、地域の特性に合わせた施策を検討することが重要です。', '[{"title": "地域活性化成功事例集", "url": "https://example.com/case-studies", "page": 15, "score": 0.95}]', 1);

INSERT OR IGNORE INTO audit_logs (id, project_id, step_id, session_id, coworker_id, action, detail, created_at) VALUES ('550e8400-e29b-41d4-a716-446655440080', '550e8400-e29b-41d4-a716-446655440000', '550e8400-e29b-41d4-a716-446655440010', '550e8400-e29b-41d4-a716-446655440050', 1, 'create', '{"action": "プロジェクト作成", "project_name": "地域活性化プロジェクト"}', '2024-01-15 09:00:00');
INSERT OR IGNORE INTO audit_logs (id, project_id, step_id, session_id, coworker_id, action, detail, created_at) VALUES ('550e8400-e29b-41d4-a716-446655440081', '550e8400-e29b-41d4-a716-446655440000', '550e8400-e29b-41d4-a716-446655440010', '550e8400-e29b-41d4-a716-446655440050', 1, 'update', '{"action": "ステップセクション更新", "section": "problem", "content": "地域の人口減少と高齢化が進んでいる..."}', '2024-01-15 10:30:00');

COMMIT;

-- 完了メッセージ
SELECT 'Database creation completed successfully!' AS status;
