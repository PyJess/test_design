#FROM python:3.12-slim

FROM generativeaidevagentsregistry.azurecr.io/python:3.12-slim

WORKDIR /test_design

COPY . /test_design

RUN pip install --upgrade pip && pip install -r requirements.txt

RUN apt-get update && apt-get install -y pandoc

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
