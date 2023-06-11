FROM tensorflow/tensorflow

WORKDIR /usr/src/backend-api

COPY . .

RUN [ "pip", "install", "-r", "requirements.txt" ]

CMD ["python", "smartgizi.py"]