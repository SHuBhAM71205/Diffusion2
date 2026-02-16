FROM python:3.13
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml .

RUN pip install --upgrade pip
RUN pip install .

COPY . .

EXPOSE 8000

CMD ["python", "main.py", "serve"]