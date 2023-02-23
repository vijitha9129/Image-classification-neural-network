FROM python:3.8-slim-buster

EXPOSE 8073

COPY requirements.txt .

RUN python -m pip install -r requirements.txt
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

WORKDIR /app

COPY . /app

CMD ["python","app.py"]

# if you want to new furniture.h5 file uncomment below line
#CMD python model.py && python app.py