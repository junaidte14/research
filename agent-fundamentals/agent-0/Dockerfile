FROM python:3.12-slim
RUN pip install --no-cache-dir pytest
WORKDIR /workspace

#Build it once, with internet access, from the agent-0 folder:
#docker build -t agent-sandbox .