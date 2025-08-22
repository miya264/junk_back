#!/usr/bin/env python3
"""
Detailed API debugging script
"""

import requests
import json
import traceback

def test_simple_save():
    """Test with minimal data"""
    url = "http://127.0.0.1:8000/api/project-step-sections"
    
    # First create a project
    project_data = {
        "name": "Debug Test Project",
        "description": "Test project for debugging",
        "owner_coworker_id": 1,
        "member_ids": [1]
    }
    
    try:
        print("1. Creating test project...")
        project_response = requests.post("http://127.0.0.1:8000/api/projects", json=project_data)
        print(f"Project creation status: {project_response.status_code}")
        
        if project_response.status_code != 200:
            print(f"Project creation failed: {project_response.text}")
            return
            
        project = project_response.json()
        project_id = project['id']
        print(f"Created project: {project_id}")
        
        # Now test step sections
        print("\n2. Testing step sections save...")
        data = {
            "project_id": project_id,
            "step_key": "analysis",
            "sections": [
                {
                    "section_key": "problem",
                    "content": "Test problem content",
                    "label": "Problem section"
                }
            ]
        }
        
        print(f"Request data: {json.dumps(data, indent=2)}")
        
        response = requests.post(url, json=data)
        print(f"Response status: {response.status_code}")
        print(f"Response headers: {dict(response.headers)}")
        print(f"Response content: {response.text}")
        
        if response.status_code == 200:
            print("SUCCESS: Step sections saved")
        else:
            print("FAILED: Step sections save failed")
            
    except Exception as e:
        print(f"ERROR: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    test_simple_save()