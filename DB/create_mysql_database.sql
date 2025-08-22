-- =========================================
-- Azure Database for MySQL 用 初期スキーマ（INTに統一 / UNSIGNED排除）
-- =========================================
SET NAMES utf8mb4;
SET SESSION collation_connection = 'utf8mb4_unicode_ci';
SET FOREIGN_KEY_CHECKS = 1;

-- ===== 参照先テーブルを先に作成（順序重要） =====
CREATE TABLE IF NOT EXISTS departments (
  id            INT NOT NULL AUTO_INCREMENT,
  name          VARCHAR(255) NOT NULL UNIQUE,
  PRIMARY KEY (id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS coworkers (
  id             INT NOT NULL AUTO_INCREMENT,
  name           VARCHAR(255) NOT NULL,
  position       VARCHAR(255),
  email          VARCHAR(255) NOT NULL UNIQUE,
  -- 既存に合わせて NOT NULL を付けない（NULL可・UNIQUE）
  sso_id         VARCHAR(255) UNIQUE,
  department_id  INT,
  PRIMARY KEY (id),
  KEY idx_cw_dept (department_id),
  CONSTRAINT fk_cw_dept
    FOREIGN KEY (department_id) REFERENCES departments(id)
    ON UPDATE RESTRICT ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS auth_users (
  id             INT NOT NULL AUTO_INCREMENT,
  coworker_id    INT NOT NULL UNIQUE,
  password_hash  VARCHAR(255) NOT NULL,
  last_login     TIMESTAMP NULL DEFAULT NULL,
  PRIMARY KEY (id),
  CONSTRAINT fk_auth_cw
    FOREIGN KEY (coworker_id) REFERENCES coworkers(id)
    ON UPDATE RESTRICT ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ===== プロジェクト管理 =====
CREATE TABLE IF NOT EXISTS projects (
  id                 CHAR(36) NOT NULL,
  owner_coworker_id  INT NOT NULL,
  name               VARCHAR(200) NOT NULL,
  description        TEXT,
  status             VARCHAR(40),
  created_at         TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at         TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  KEY idx_projects_owner (owner_coworker_id),
  CONSTRAINT fk_projects_owner
    FOREIGN KEY (owner_coworker_id) REFERENCES coworkers(id)
    ON UPDATE RESTRICT ON DELETE RESTRICT
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS project_members (
  id           CHAR(36) NOT NULL,
  project_id   CHAR(36) NOT NULL,
  coworker_id  INT NOT NULL,
  role         VARCHAR(40),
  joined_at    TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  UNIQUE KEY uidx_pm_project_cw (project_id, coworker_id),
  KEY idx_pm_project (project_id),
  KEY idx_pm_coworker (coworker_id),
  CONSTRAINT fk_pm_project
    FOREIGN KEY (project_id) REFERENCES projects(id)
    ON UPDATE RESTRICT ON DELETE CASCADE,
  CONSTRAINT fk_pm_coworker
    FOREIGN KEY (coworker_id) REFERENCES coworkers(id)
    ON UPDATE RESTRICT ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ===== ステップ定義 & 実体 =====
CREATE TABLE IF NOT EXISTS policy_steps (
  id         CHAR(36) NOT NULL,
  project_id CHAR(36) NOT NULL,
  step_key   VARCHAR(60) NOT NULL,
  step_name  VARCHAR(120) NOT NULL,
  order_no   INT NOT NULL,
  status     VARCHAR(40),
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  UNIQUE KEY uidx_ps_project_stepkey (project_id, step_key),
  KEY idx_ps_project (project_id),
  CONSTRAINT fk_ps_project
    FOREIGN KEY (project_id) REFERENCES projects(id)
    ON UPDATE RESTRICT ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS step_templates (
  id         CHAR(36) NOT NULL,
  step_key   VARCHAR(60) NOT NULL,
  title      TEXT NOT NULL,
  version    INT NOT NULL DEFAULT 1,
  is_active  TINYINT(1) NOT NULL DEFAULT 1,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  UNIQUE KEY uidx_st_stepkey_version (step_key, version)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS step_template_sections (
  id                  CHAR(36) NOT NULL,
  template_id         CHAR(36) NOT NULL,
  order_no            INT NOT NULL,
  section_key         VARCHAR(50),
  label               TEXT NOT NULL,
  field_type          VARCHAR(30) NOT NULL DEFAULT 'text',
  options_json        TEXT,
  validation_schema   TEXT,
  default_placeholder TEXT,
  default_help        TEXT,
  is_required         TINYINT(1) NOT NULL DEFAULT 0,
  created_at          TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at          TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  UNIQUE KEY uidx_sts_template_order (template_id, order_no),
  KEY idx_sts_template (template_id),
  CONSTRAINT fk_sts_template
    FOREIGN KEY (template_id) REFERENCES step_templates(id)
    ON UPDATE RESTRICT ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS project_step_sections (
  id                  CHAR(36) NOT NULL,
  project_id          CHAR(36) NOT NULL,
  step_id             CHAR(36) NOT NULL,
  template_section_id CHAR(36),
  order_no            INT NOT NULL,
  section_key         VARCHAR(50),
  label               TEXT NOT NULL,
  field_type          VARCHAR(30) NOT NULL DEFAULT 'text',
  content_text        LONGTEXT,
  content_json        LONGTEXT,
  is_required         TINYINT(1),
  created_by          INT,
  created_at          TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at          TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  UNIQUE KEY uidx_pss_step_order (step_id, order_no),
  KEY idx_pss_project (project_id),
  KEY idx_pss_step (step_id),
  CONSTRAINT fk_pss_project
    FOREIGN KEY (project_id) REFERENCES projects(id)
    ON UPDATE RESTRICT ON DELETE CASCADE,
  CONSTRAINT fk_pss_step
    FOREIGN KEY (step_id) REFERENCES policy_steps(id)
    ON UPDATE RESTRICT ON DELETE CASCADE,
  CONSTRAINT fk_pss_template_section
    FOREIGN KEY (template_section_id) REFERENCES step_template_sections(id)
    ON UPDATE RESTRICT ON DELETE SET NULL,
  CONSTRAINT fk_pss_created_by
    FOREIGN KEY (created_by) REFERENCES coworkers(id)
    ON UPDATE RESTRICT ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ===== チャット =====
CREATE TABLE IF NOT EXISTS chat_sessions (
  id          CHAR(36) NOT NULL,
  project_id  CHAR(36) NOT NULL,
  step_id     CHAR(36),
  title       VARCHAR(200),
  created_by  INT,
  created_at  TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at  TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  KEY idx_cs_project (project_id),
  KEY idx_cs_step (step_id),
  CONSTRAINT fk_cs_project
    FOREIGN KEY (project_id) REFERENCES projects(id)
    ON UPDATE RESTRICT ON DELETE CASCADE,
  CONSTRAINT fk_cs_step
    FOREIGN KEY (step_id) REFERENCES policy_steps(id)
    ON UPDATE RESTRICT ON DELETE SET NULL,
  CONSTRAINT fk_cs_created_by
    FOREIGN KEY (created_by) REFERENCES coworkers(id)
    ON UPDATE RESTRICT ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS chat_messages (
  id          CHAR(36) NOT NULL,
  session_id  CHAR(36) NOT NULL,
  project_id  CHAR(36) NOT NULL,
  step_id     CHAR(36),
  role        VARCHAR(20) NOT NULL,
  msg_type    VARCHAR(20) NOT NULL,
  content     LONGTEXT NOT NULL,
  content_json LONGTEXT,
  created_at  TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  KEY idx_cm_session (session_id),
  KEY idx_cm_project_step (project_id, step_id),
  KEY idx_cm_type (msg_type),
  CONSTRAINT fk_cm_session
    FOREIGN KEY (session_id) REFERENCES chat_sessions(id)
    ON UPDATE RESTRICT ON DELETE CASCADE,
  CONSTRAINT fk_cm_project
    FOREIGN KEY (project_id) REFERENCES projects(id)
    ON UPDATE RESTRICT ON DELETE CASCADE,
  CONSTRAINT fk_cm_step
    FOREIGN KEY (step_id) REFERENCES policy_steps(id)
    ON UPDATE RESTRICT ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ===== RAG =====
CREATE TABLE IF NOT EXISTS rag_search_results (
  id           CHAR(36) NOT NULL,
  project_id   CHAR(36) NOT NULL,
  step_id      CHAR(36) NOT NULL,
  session_id   CHAR(36),
  query        TEXT NOT NULL,
  result_text  LONGTEXT,
  result_json  LONGTEXT,
  sources_json LONGTEXT,
  created_by   INT,
  created_at   TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  KEY idx_rsr_project_step (project_id, step_id),
  KEY idx_rsr_session (session_id),
  KEY idx_rsr_created (created_at),
  CONSTRAINT fk_rsr_project
    FOREIGN KEY (project_id) REFERENCES projects(id)
    ON UPDATE RESTRICT ON DELETE CASCADE,
  CONSTRAINT fk_rsr_step
    FOREIGN KEY (step_id) REFERENCES policy_steps(id)
    ON UPDATE RESTRICT ON DELETE CASCADE,
  CONSTRAINT fk_rsr_session
    FOREIGN KEY (session_id) REFERENCES chat_sessions(id)
    ON UPDATE RESTRICT ON DELETE SET NULL,
  CONSTRAINT fk_rsr_created_by
    FOREIGN KEY (created_by) REFERENCES coworkers(id)
    ON UPDATE RESTRICT ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS rag_result_sources (
  id             CHAR(36) NOT NULL,
  rag_result_id  CHAR(36) NOT NULL,
  source_title   VARCHAR(300),
  source_url     VARCHAR(500),
  section_title  VARCHAR(300),
  page_number    INT,
  year_label     VARCHAR(20),
  score          DOUBLE,
  PRIMARY KEY (id),
  KEY idx_rrs_result (rag_result_id),
  CONSTRAINT fk_rrs_result
    FOREIGN KEY (rag_result_id) REFERENCES rag_search_results(id)
    ON UPDATE RESTRICT ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ===== 監査ログ =====
CREATE TABLE IF NOT EXISTS audit_logs (
  id          CHAR(36) NOT NULL,
  project_id  CHAR(36),
  step_id     CHAR(36),
  session_id  CHAR(36),
  coworker_id INT,
  action      VARCHAR(50),
  detail      LONGTEXT,
  created_at  TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  KEY idx_al_project (project_id),
  KEY idx_al_step (step_id),
  KEY idx_al_session (session_id),
  KEY idx_al_coworker (coworker_id),
  KEY idx_al_action (action),
  KEY idx_al_created (created_at),
  CONSTRAINT fk_al_project
    FOREIGN KEY (project_id) REFERENCES projects(id)
    ON UPDATE RESTRICT ON DELETE SET NULL,
  CONSTRAINT fk_al_step
    FOREIGN KEY (step_id) REFERENCES policy_steps(id)
    ON UPDATE RESTRICT ON DELETE SET NULL,
  CONSTRAINT fk_al_session
    FOREIGN KEY (session_id) REFERENCES chat_sessions(id)
    ON UPDATE RESTRICT ON DELETE SET NULL,
  CONSTRAINT fk_al_coworker
    FOREIGN KEY (coworker_id) REFERENCES coworkers(id)
    ON UPDATE RESTRICT ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- =========================================
-- サンプル初期データ（必要に応じて実行）
-- =========================================
START TRANSACTION;

-- プロジェクト
INSERT INTO projects (id, owner_coworker_id, name, description, status)
VALUES 
('550e8400-e29b-41d4-a716-446655440000', 1, '地域活性化プロジェクト', '地域経済の活性化を目指す政策立案プロジェクト', 'active')
AS new
ON DUPLICATE KEY UPDATE
  name=new.name, description=new.description, status=new.status, updated_at=CURRENT_TIMESTAMP;

INSERT INTO projects (id, owner_coworker_id, name, description, status)
VALUES 
('550e8400-e29b-41d4-a716-446655440001', 2, 'デジタル化推進', '行政サービスのデジタル化推進プロジェクト', 'active')
AS new
ON DUPLICATE KEY UPDATE
  name=new.name, description=new.description, status=new.status, updated_at=CURRENT_TIMESTAMP;

-- プロジェクトメンバー（オーナーを紐づけ）
INSERT IGNORE INTO project_members (id, project_id, coworker_id, role)
VALUES
('11111111-1111-1111-1111-111111111111','550e8400-e29b-41d4-a716-446655440000',1,'owner'),
('22222222-2222-2222-2222-222222222222','550e8400-e29b-41d4-a716-446655440001',2,'owner');

-- ステップ（プロジェクト: 0000）
INSERT INTO policy_steps (id, project_id, step_key, step_name, order_no, status)
VALUES
('550e8400-e29b-41d4-a716-446655440010','550e8400-e29b-41d4-a716-446655440000','analysis','現状分析・課題整理',1,'in_progress'),
('550e8400-e29b-41d4-a716-446655440011','550e8400-e29b-41d4-a716-446655440000','objective','目的整理',2,'draft'),
('550e8400-e29b-41d4-a716-446655440012','550e8400-e29b-41d4-a716-446655440000','concept','コンセプト策定',3,'draft'),
('550e8400-e29b-41d4-a716-446655440013','550e8400-e29b-41d4-a716-446655440000','plan','施策案作成',4,'draft'),
('550e8400-e29b-41d4-a716-446655440014','550e8400-e29b-41d4-a716-446655440000','proposal','提案書作成',5,'draft')
AS new
ON DUPLICATE KEY UPDATE
  step_name=new.step_name, order_no=new.order_no, status=new.status, updated_at=CURRENT_TIMESTAMP;

-- テンプレート
INSERT INTO step_templates (id, step_key, title, version, is_active)
VALUES
('550e8400-e29b-41d4-a716-446655440020','analysis','現状分析・課題整理',1,1),
('550e8400-e29b-41d4-a716-446655440021','objective','目的整理',1,1),
('550e8400-e29b-41d4-a716-446655440022','concept','コンセプト策定',1,1),
('550e8400-e29b-41d4-a716-446655440023','plan','施策案作成',1,1),
('550e8400-e29b-41d4-a716-446655440024','proposal','提案書作成',1,1)
AS new
ON DUPLICATE KEY UPDATE
  title=new.title, is_active=new.is_active, updated_at=CURRENT_TIMESTAMP;

-- テンプレート・セクション（UPSERT：template_id+order_no で一意）
-- ANALYSIS
INSERT INTO step_template_sections
(id, template_id, order_no, section_key, label, field_type, default_placeholder, default_help, is_required)
VALUES
('550e8400-e29b-41d4-a716-446655440030','550e8400-e29b-41d4-a716-446655440020',1,'problem','課題と裏付け','richtext','統計/調査/事例の根拠','課題と裏付け（定量・定性）を記入してください',1)
AS new
ON DUPLICATE KEY UPDATE
  section_key=new.section_key, label=new.label, field_type=new.field_type,
  default_placeholder=new.default_placeholder, default_help=new.default_help,
  is_required=new.is_required, updated_at=CURRENT_TIMESTAMP;

INSERT INTO step_template_sections
(id, template_id, order_no, section_key, label, field_type, default_placeholder, default_help, is_required)
VALUES
('550e8400-e29b-41d4-a716-446655440031','550e8400-e29b-41d4-a716-446655440020',2,'background','背景構造の評価','text','制度・市場・競合…','課題の背景にある構造（制度・市場など）を簡単に評価してください',1)
AS new
ON DUPLICATE KEY UPDATE
  section_key=new.section_key, label=new.label, field_type=new.field_type,
  default_placeholder=new.default_placeholder, default_help=new.default_help,
  is_required=new.is_required, updated_at=CURRENT_TIMESTAMP;

INSERT INTO step_template_sections
(id, template_id, order_no, section_key, label, field_type, default_placeholder, default_help, is_required)
VALUES
('550e8400-e29b-41d4-a716-446655440032','550e8400-e29b-41d4-a716-446655440020',3,'priority','優先度と理由','text','優先度と根拠','解決すべき課題の優先度と理由を整理しましょう',1)
AS new
ON DUPLICATE KEY UPDATE
  section_key=new.section_key, label=new.label, field_type=new.field_type,
  default_placeholder=new.default_placeholder, default_help=new.default_help,
  is_required=new.is_required, updated_at=CURRENT_TIMESTAMP;

-- OBJECTIVE
INSERT INTO step_template_sections
(id, template_id, order_no, section_key, label, field_type, default_placeholder, default_help, is_required)
VALUES
('550e8400-e29b-41d4-a716-446655440040','550e8400-e29b-41d4-a716-446655440021',1,'goal','最終ゴール','text','具体的な到達像','最終的に達成したいゴールを具体的に記載してください',1)
AS new
ON DUPLICATE KEY UPDATE
  section_key=new.section_key, label=new.label, field_type=new.field_type,
  default_placeholder=new.default_placeholder, default_help=new.default_help,
  is_required=new.is_required, updated_at=CURRENT_TIMESTAMP;

INSERT INTO step_template_sections
(id, template_id, order_no, section_key, label, field_type, default_placeholder, default_help, is_required)
VALUES
('550e8400-e29b-41d4-a716-446655440041','550e8400-e29b-41d4-a716-446655440021',2,'kpi','KPI・目標値','text','いつまでに/どれだけ','KPI・目標値（いつまでに・どれだけ）を記入してください',1)
AS new
ON DUPLICATE KEY UPDATE
  section_key=new.section_key, label=new.label, field_type=new.field_type,
  default_placeholder=new.default_placeholder, default_help=new.default_help,
  is_required=new.is_required, updated_at=CURRENT_TIMESTAMP;

INSERT INTO step_template_sections
(id, template_id, order_no, section_key, label, field_type, default_placeholder, default_help, is_required)
VALUES
('550e8400-e29b-41d4-a716-446655440042','550e8400-e29b-41d4-a716-446655440021',3,'constraints','前提条件・制約','text','予算/人員/期間','前提条件・制約（予算、人員、期間など）を整理しましょう',0)
AS new
ON DUPLICATE KEY UPDATE
  section_key=new.section_key, label=new.label, field_type=new.field_type,
  default_placeholder=new.default_placeholder, default_help=new.default_help,
  is_required=new.is_required, updated_at=CURRENT_TIMESTAMP;

-- CONCEPT
INSERT INTO step_template_sections
(id, template_id, order_no, section_key, label, field_type, default_placeholder, default_help, is_required)
VALUES
('550e8400-e29b-41d4-a716-446655440050','550e8400-e29b-41d4-a716-446655440022',1,'policy','基本方針','richtext','価値/対象/提供方法','基本方針（どんな価値を誰に、どう届けるか）を記入してください',1)
AS new
ON DUPLICATE KEY UPDATE
  section_key=new.section_key, label=new.label, field_type=new.field_type,
  default_placeholder=new.default_placeholder, default_help=new.default_help,
  is_required=new.is_required, updated_at=CURRENT_TIMESTAMP;

INSERT INTO step_template_sections
(id, template_id, order_no, section_key, label, field_type, default_placeholder, default_help, is_required)
VALUES
('550e8400-e29b-41d4-a716-446655440051','550e8400-e29b-41d4-a716-446655440022',2,'rationale','根拠・示唆','text','調査/事例/専門家意見','方針の根拠・示唆（調査、事例、専門家意見など）を書いてください',0)
AS new
ON DUPLICATE KEY UPDATE
  section_key=new.section_key, label=new.label, field_type=new.field_type,
  default_placeholder=new.default_placeholder, default_help=new.default_help,
  is_required=new.is_required, updated_at=CURRENT_TIMESTAMP;

INSERT INTO step_template_sections
(id, template_id, order_no, section_key, label, field_type, default_placeholder, default_help, is_required)
VALUES
('550e8400-e29b-41d4-a716-446655440052','550e8400-e29b-41d4-a716-446655440022',3,'risks','リスクと打ち手','text','代替案/実験設計','主要リスクと打ち手（代替案、実験設計）を整理しましょう',0)
AS new
ON DUPLICATE KEY UPDATE
  section_key=new.section_key, label=new.label, field_type=new.field_type,
  default_placeholder=new.default_placeholder, default_help=new.default_help,
  is_required=new.is_required, updated_at=CURRENT_TIMESTAMP;

-- PLAN
INSERT INTO step_template_sections
(id, template_id, order_no, section_key, label, field_type, default_placeholder, default_help, is_required)
VALUES
('550e8400-e29b-41d4-a716-446655440060','550e8400-e29b-41d4-a716-446655440023',1,'initiatives','主な施策','text','3〜5施策の概要と狙い','主な施策（3〜5個）の概要と狙いを整理してください',1)
AS new
ON DUPLICATE KEY UPDATE
  section_key=new.section_key, label=new.label, field_type=new.field_type,
  default_placeholder=new.default_placeholder, default_help=new.default_help,
  is_required=new.is_required, updated_at=CURRENT_TIMESTAMP;

INSERT INTO step_template_sections
(id, template_id, order_no, section_key, label, field_type, default_placeholder, default_help, is_required)
VALUES
('550e8400-e29b-41d4-a716-446655440061','550e8400-e29b-41d4-a716-446655440023',2,'schedule','体制・スケジュール','text','役割分担/マイルストーン','体制・役割分担・スケジュールを記入してください',0)
AS new
ON DUPLICATE KEY UPDATE
  section_key=new.section_key, label=new.label, field_type=new.field_type,
  default_placeholder=new.default_placeholder, default_help=new.default_help,
  is_required=new.is_required, updated_at=CURRENT_TIMESTAMP;

INSERT INTO step_template_sections
(id, template_id, order_no, section_key, label, field_type, default_placeholder, default_help, is_required)
VALUES
('550e8400-e29b-41d4-a716-446655440062','550e8400-e29b-41d4-a716-446655440023',3,'cost','コストと効果','text','概算コスト/効果/根拠','概算コスト・効果見込み（根拠も）を書いてください',0)
AS new
ON DUPLICATE KEY UPDATE
  section_key=new.section_key, label=new.label, field_type=new.field_type,
  default_placeholder=new.default_placeholder, default_help=new.default_help,
  is_required=new.is_required, updated_at=CURRENT_TIMESTAMP;

-- PROPOSAL
INSERT INTO step_template_sections
(id, template_id, order_no, section_key, label, field_type, default_placeholder, default_help, is_required)
VALUES
('550e8400-e29b-41d4-a716-446655440070','550e8400-e29b-41d4-a716-446655440024',1,'summary','提案サマリー','richtext','背景→課題→解決→効果→体制','提案のサマリー（背景→課題→解決→効果→体制）を書いてください',1)
AS new
ON DUPLICATE KEY UPDATE
  section_key=new.section_key, label=new.label, field_type=new.field_type,
  default_placeholder=new.default_placeholder, default_help=new.default_help,
  is_required=new.is_required, updated_at=CURRENT_TIMESTAMP;

INSERT INTO step_template_sections
(id, template_id, order_no, section_key, label, field_type, default_placeholder, default_help, is_required)
VALUES
('550e8400-e29b-41d4-a716-446655440071','550e8400-e29b-41d4-a716-446655440024',2,'concerns','意思決定者の関心','text','費用対効果/リスク/責任分担','意思決定者の関心（費用対効果、リスク、責任分担）を整理してください',0)
AS new
ON DUPLICATE KEY UPDATE
  section_key=new.section_key, label=new.label, field_type=new.field_type,
  default_placeholder=new.default_placeholder, default_help=new.default_help,
  is_required=new.is_required, updated_at=CURRENT_TIMESTAMP;

INSERT INTO step_template_sections
(id, template_id, order_no, section_key, label, field_type, default_placeholder, default_help, is_required)
VALUES
('550e8400-e29b-41d4-a716-446655440072','550e8400-e29b-41d4-a716-446655440024',3,'next_steps','次のアクション','text','承認/説明/PoC準備','次のアクション（承認プロセス、関係者説明、PoC準備など）を記入してください',0)
AS new
ON DUPLICATE KEY UPDATE
  section_key=new.section_key, label=new.label, field_type=new.field_type,
  default_placeholder=new.default_placeholder, default_help=new.default_help,
  is_required=new.is_required, updated_at=CURRENT_TIMESTAMP;

COMMIT;

-- 動作確認用
SELECT 'Database creation completed successfully! (MySQL)' AS status;
