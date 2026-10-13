FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .

RUN python -m pip install --upgrade pip

RUN python -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
RUN python -m pip install --verbose --timeout 30 --retries 2 -r requirements.txt

COPY . .

EXPOSE 8000 8501

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]