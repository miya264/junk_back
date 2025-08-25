#!/usr/bin/env python3
"""
テストデータ用のパスワードハッシュを生成するスクリプト
"""

from passlib.hash import bcrypt

# テストユーザーのパスワード
test_passwords = {
    "tanaka@example.com": "tanaka123",
    "sato@example.com": "sato123", 
    "suzuki@example.com": "suzuki123"
}

print("テストユーザーのパスワードハッシュ:")
print("=" * 50)

for email, password in test_passwords.items():
    hashed = bcrypt.hash(password)
    print(f"Email: {email}")
    print(f"Password: {password}")
    print(f"Hash: {hashed}")
    print("-" * 30)

print("\nSQLアップデート文:")
print("=" * 50)

emails_to_coworker_id = {
    "tanaka@example.com": 1,
    "sato@example.com": 2,
    "suzuki@example.com": 3
}

for email, password in test_passwords.items():
    hashed = bcrypt.hash(password)
    coworker_id = emails_to_coworker_id[email]
    print(f"UPDATE auth_users SET password_hash = '{hashed}' WHERE coworker_id = {coworker_id};")