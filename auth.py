from pydantic import BaseModel
from typing import Optional, Dict, Any
from fastapi import Response
import uuid
from datetime import datetime, timedelta
import jwt
import os
from passlib.context import CryptContext

class LoginRequest(BaseModel):
    username: Optional[str] = None
    password: Optional[str] = None
    user: Optional[str] = None  # Alternative field name
    pass_: Optional[str] = None  # Alternative field name (for pass)
    passwd: Optional[str] = None  # Alternative field name
    pwd: Optional[str] = None    # Alternative field name  
    email: Optional[str] = None  # Alternative field name
    
    def get_username(self) -> str:
        return self.username or self.user or self.email or ""
    
    def get_password(self) -> str:
        return self.password or self.pass_ or self.passwd or self.pwd or ""

class LoginResponse(BaseModel):
    access_token: str
    token_type: str
    user: Dict[str, Any]

class AuthService:
    def __init__(self):
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        self.secret_key = os.getenv("JWT_SECRET_KEY", "default-secret-key-change-in-production")
        self.algorithm = "HS256"
        self.access_token_expire_minutes = 30

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        return self.pwd_context.verify(plain_password, hashed_password)

    def get_password_hash(self, password: str) -> str:
        return self.pwd_context.hash(password)

    def create_access_token(self, data: dict) -> str:
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt

    async def authenticate_user(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        try:
            # まずハードコードされたadminユーザーをチェック
            if username == "admin" and password == "admin":
                return {
                    "id": 1,
                    "username": username,
                    "email": f"{username}@example.com",
                    "is_active": True
                }
            
            # データベース接続を試行
            try:
                from DB.mysql.mysql_connection import get_mysql_db
                db = get_mysql_db()
                
                # データベースからユーザーを検索（メールまたはユーザー名で）
                query = """
                    SELECT au.id, au.coworker_id, au.password_hash, au.last_login,
                           c.name, c.email, c.position, c.sso_id, d.name as department_name
                    FROM auth_users au
                    JOIN coworkers c ON au.coworker_id = c.id
                    LEFT JOIN departments d ON c.department_id = d.id
                    WHERE c.email = %s OR c.name = %s
                """
                result = await db.execute_query(query, (username, username))
                
                if not result:
                    print(f"User not found in database: {username}")
                    return None
                
                user = result[0]
                
                # パスワードの検証
                if self.verify_password(password, user.get('password_hash', '')):
                    return {
                        "id": user['coworker_id'],  # Use coworker_id as the main ID
                        "username": user['name'],   # Use name as username
                        "email": user['email'],
                        "position": user.get('position', ''),
                        "department_name": user.get('department_name', ''),
                        "is_active": True  # Assume active if in the system
                    }
                else:
                    print(f"Password verification failed for user: {username}")
                    return None
                    
            except Exception as db_error:
                print(f"Database authentication error: {db_error}")
                # データベースエラーの場合、テスト用ユーザーで認証を試行
                test_users = {
                    "ishigami@example.com": "password123",
                    "test@example.com": "password123"
                }
                
                if username in test_users and password == test_users[username]:
                    return {
                        "id": 999,
                        "username": username.split('@')[0],
                        "email": username,
                        "is_active": True
                    }
                return None
                
        except Exception as e:
            print(f"Authentication error: {e}")
            return None

    async def login(self, request: LoginRequest, response: Response) -> LoginResponse:
        # Debug logging
        print(f"Login request received: {request.dict()}")
        
        username = request.get_username()
        password = request.get_password()
        
        print(f"Extracted username: {username}, password: {'***' if password else 'None'}")
        
        if not username or not password:
            from fastapi import HTTPException
            raise HTTPException(status_code=400, detail="Username and password are required")
        
        user = await self.authenticate_user(username, password)
        if not user:
            from fastapi import HTTPException
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        access_token = self.create_access_token(data={"sub": user["username"], "user_id": user["id"]})
        
        # HTTPOnlyクッキーとしてトークンを設定
        response.set_cookie(
            key="access_token",
            value=access_token,
            httponly=True,
            max_age=self.access_token_expire_minutes * 60,
            samesite="lax"
        )
        
        return LoginResponse(
            access_token=access_token,
            token_type="bearer",
            user=user
        )

    async def logout(self, response: Response):
        response.delete_cookie(key="access_token")

    async def get_current_user(self, access_token: Optional[str]) -> Optional[Dict[str, Any]]:
        if not access_token:
            return None
        
        try:
            payload = jwt.decode(access_token, self.secret_key, algorithms=[self.algorithm])
            username: str = payload.get("sub")
            user_id: int = payload.get("user_id")
            if username is None or user_id is None:
                return None
            
            # 実際の本番環境では、ここでデータベースからユーザー情報を取得
            return {
                "id": user_id,
                "username": username,
                "email": f"{username}@example.com",
                "is_active": True
            }
        except jwt.PyJWTError:
            return None

# シングルトンインスタンス
auth_service = AuthService()