FROM python:3.12-slim
WORKDIR /app
COPY requirements-lock.txt .
RUN pip install --no-cache-dir -r requirements-lock.txt
COPY src ./src
COPY app ./app
COPY reports ./reports
# The dataset is fetched at build time so the image is self-contained.
RUN python src/download_data.py && python src/data_prep.py
ENV MPLBACKEND=Agg
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
