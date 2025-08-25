-- テストデータ用の部署とcoworkerを挿入
INSERT IGNORE INTO departments (id, name) VALUES 
(1, '政策企画部'),
(2, 'デジタル推進課');

INSERT IGNORE INTO coworkers (id, name, position, email, sso_id, department_id) VALUES 
(1, '田中 太郎', '課長', 'tanaka@example.com', 'tanaka001', 1),
(2, '佐藤 花子', '主任', 'sato@example.com', 'sato002', 1), 
(3, '鈴木 次郎', '係長', 'suzuki@example.com', 'suzuki003', 2);

-- テストユーザーの認証情報（実際のbcryptハッシュを使用）
-- パスワード: tanaka123, sato123, suzuki123
INSERT IGNORE INTO auth_users (id, coworker_id, password_hash, last_login) VALUES 
(1001, 1, '$2b$12$yMcKHiJu/E9ufJpKIFSm/OeHmUeCmagXAZUYROXKQEil6sqIGLFam', NULL),
(1002, 2, '$2b$12$FqWuT5IVkKcOnIijV9P6FeUZGjrIqKoLCXQUyVZjgUqP5D0Y39g7O', NULL),
(1003, 3, '$2b$12$mjatF22vo2OLHX3eFHf.heeubc0xXsCHDbnU8hPiRgkcUZ9QDjQoa', NULL);

-- 確認
SELECT 
    c.id,
    c.name,
    c.email,
    c.position,
    d.name as department_name,
    CASE WHEN au.coworker_id IS NOT NULL THEN 'Yes' ELSE 'No' END as has_auth
FROM coworkers c
LEFT JOIN departments d ON c.department_id = d.id
LEFT JOIN auth_users au ON au.coworker_id = c.id
ORDER BY c.id;