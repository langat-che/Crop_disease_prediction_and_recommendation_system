services:
  - type: web
    name: crop-disease-api
    env: python
    region: iad
    plan: free
    branch: main
    buildCommand: pip install -r backend/requirements.txt
    startCommand: python backend/app_lite.py
    envVars:
      - key: PORT
        value: "10000"
