FROM python:3.9.6

WORKDIR /code

EXPOSE 9000

COPY requirements.txt .
COPY setup.py .
COPY PopSynthesis ./PopSynthesis

RUN pip install -r requirements.txt
RUN pip install -e .

CMD ["python", "PopSynthesis/main.py"]