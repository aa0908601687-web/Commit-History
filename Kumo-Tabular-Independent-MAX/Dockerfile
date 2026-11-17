FROM python:3.11-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY requirements.txt requirements-production.txt ./
RUN python -m pip install --no-cache-dir -r requirements-production.txt
COPY . .
RUN useradd --create-home --uid 10001 appuser
USER appuser
EXPOSE 8000
CMD ["uvicorn","production_api:app","--host","0.0.0.0","--port","8000"]
