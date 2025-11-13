FROM python:3.9.19-slim
WORKDIR /marmoset_classification
COPY . /marmoset_classification
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 5000
CMD ["python", "app.py"]