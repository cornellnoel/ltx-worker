FROM runpod/base:0.6.3-cuda11.8.0

RUN pip install runpod

COPY handler_minimal.py /handler.py

CMD ["python", "-u", "/handler.py"]
