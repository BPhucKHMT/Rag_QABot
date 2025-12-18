FROM python:3.12.7
LABEL author="Bao Phuc"

WORKDIR  /src


COPY requirements.txt /src/requirements.txt

RUN pip install --no-cache-dir -r /src/requirements.txt

#copy project
COPY . /src

# default command: chạy uvicorn server (docker-compose override để chạy streamlit)
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

