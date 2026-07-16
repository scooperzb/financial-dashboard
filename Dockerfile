FROM python:3.12-slim

WORKDIR /app

# Install dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app files
COPY app.py holdings.json block_trades.json ./
COPY .streamlit/ .streamlit/
COPY models/ models/
COPY pages/ pages/
COPY snapshots/ snapshots/

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
