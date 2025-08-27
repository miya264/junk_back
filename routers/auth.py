from fastapi import APIRouter, HTTPException, Response, Cookie
from pydantic import BaseModel
from typing import Optional, Dict, Any
from auth import auth_service, LoginRequest, LoginResponse

router = APIRouter(prefix="/api/auth", tags=["auth"])

class LoginResponseModel(LoginResponse):
    pass

@router.post("/login", response_model=LoginResponseModel)
async def login_endpoint(request: LoginRequest, response: Response):
    try:
        return await auth_service.login(request, response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/logout")
async def logout_endpoint(response: Response):
    try:
        await auth_service.logout(response)
        return {"message": "Successfully logged out"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/me")
async def get_current_user_endpoint(access_token: str = Cookie(None)):
    try:
        user = await auth_service.get_current_user(access_token)
        if not user:
            raise HTTPException(status_code=401, detail="認証が必要です")
        return user
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/verify")
async def verify_token_endpoint(access_token: str = Cookie(None)):
    try:
        user = await auth_service.get_current_user(access_token)
        if not user:
            return {"valid": False}
        return {"valid": True, "user": user}
    except Exception as e:
        return {"valid": False, "error": str(e)}