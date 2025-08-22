#!/usr/bin/env python3
"""
Test the exact same request format that frontend sends
"""

import requests
import json

def test_frontend_format():
    """Test with the exact format frontend uses"""
    
    # First create a project
    print("1. Creating project...")
    project_response = requests.post("http://127.0.0.1:8000/api/projects", json={
        "name": "Frontend Format Test",
        "description": "Test with frontend format",
        "owner_coworker_id": 1,
        "member_ids": [1]
    })
    
    if project_response.status_code != 200:
        print(f"Project creation failed: {project_response.text}")
        return
        
    project_id = project_response.json()['id']
    print(f"Created project: {project_id}")
    
    # Test the exact format frontend sends
    print("\n2. Testing with frontend format...")
    
    # This matches ContentOrganizer's format exactly
    data = {
        "project_id": project_id,
        "step_key": "analysis",
        "sections": [
            {
                "section_key": "problem",
                "content": "課題と裏付けのテスト内容",
                "label": "課題と裏付け（定量・定性）を記入してください"
            },
            {
                "section_key": "background", 
                "content": "背景構造の評価のテスト内容",
                "label": "課題の背景にある構造（制度・市場など）を簡単に評価してください"
            },
            {
                "section_key": "priority",
                "content": "優先度と理由のテスト内容",
                "label": "解決すべき課題の優先度と理由を整理しましょう"
            }
        ]
    }
    
    print(f"Request data:\n{json.dumps(data, ensure_ascii=False, indent=2)}")
    
    try:
        response = requests.post(
            "http://127.0.0.1:8000/api/project-step-sections",
            json=data,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"\nStatus Code: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code == 200:
            print("SUCCESS!")
        else:
            print("FAILED!")
            
    except Exception as e:
        print(f"ERROR: {str(e)}")

if __name__ == "__main__":
    test_frontend_format()