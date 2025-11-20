Démarrer le micro-service en mode développeur:
    
    cd /home/ubuntu/projects/anonymizer/service
    source .venv/bin/activate
    uvicorn main:app --reload
    
Test/production:
    gunicorn main:app --workers 4 --bind 0.0.0.0:8000 --worker-class uvicorn.workers.UvicornWorker