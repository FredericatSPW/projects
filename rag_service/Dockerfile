FROM python:3.12-slim

WORKDIR /app

# Copier les fichiers
COPY requirements.txt ./
COPY main.py ./

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Exposer le port FastAPI
EXPOSE 8000

# Commande pour lancer le service
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
