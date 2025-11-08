FROM python:3.10-slim-buster

WORKDIR /app


COPY . /app

# Install dependencies
RUN pip install -r requirements.txt



# Run the app
CMD ["python3", "app.py"]
