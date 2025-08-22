#!/usr/bin/env python3
"""
API endpoints test script
"""

import requests
import json

def create_test_project():
    """Create a test project for testing"""
    url = "http://127.0.0.1:8000/api/projects"
    
    data = {
        "name": "Test Project",
        "description": "Test project for API testing",
        "owner_coworker_id": 1,
        "member_ids": [1]
    }
    
    try:
        print("Creating test project...")
        response = requests.post(url, json=data, headers={"Content-Type": "application/json"})
        
        if response.status_code == 200:
            project = response.json()
            print(f"Test project created: {project['id']}")
            return project['id']
        else:
            print(f"Failed to create project: {response.text}")
            return None
            
    except Exception as e:
        print(f"Error creating project: {str(e)}")
        return None

def test_project_step_sections_save(project_id):
    """Test project step sections save endpoint"""
    url = "http://127.0.0.1:8000/api/project-step-sections"
    
    data = {
        "project_id": project_id,
        "step_key": "analysis",
        "sections": [
            {
                "section_key": "problem",
                "content": "テスト課題の内容",
                "label": "課題と裏付け"
            },
            {
                "section_key": "background", 
                "content": "テスト背景の内容",
                "label": "背景構造の評価"
            },
            {
                "section_key": "priority",
                "content": "テスト優先度の内容", 
                "label": "優先度と理由"
            }
        ]
    }
    
    try:
        print("Testing POST /api/project-step-sections...")
        print(f"Request data: {json.dumps(data, ensure_ascii=False, indent=2)}")
        
        response = requests.post(url, json=data, headers={"Content-Type": "application/json"})
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code == 200:
            print("SUCCESS: Project step sections saved")
            return True
        else:
            print("FAILED: API returned error")
            return False
            
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return False

def test_project_step_sections_get(project_id):
    """Test project step sections get endpoint"""
    step_key = "analysis"
    url = f"http://127.0.0.1:8000/api/project-step-sections/{project_id}/{step_key}"
    
    try:
        print(f"\nTesting GET /api/project-step-sections/{project_id}/{step_key}...")
        
        response = requests.get(url)
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code == 200:
            print("SUCCESS: Project step sections retrieved")
            return True
        else:
            print("FAILED: API returned error")
            return False
            
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return False

if __name__ == "__main__":
    print("=== API Endpoint Test ===\n")
    
    # Create test project first
    project_id = create_test_project()
    if not project_id:
        print("Failed to create test project, aborting tests")
        exit(1)
    
    # Test save
    save_success = test_project_step_sections_save(project_id)
    
    # Test get
    get_success = test_project_step_sections_get(project_id)
    
    print(f"\n=== Results ===")
    print(f"Save API: {'PASS' if save_success else 'FAIL'}")
    print(f"Get API: {'PASS' if get_success else 'FAIL'}")