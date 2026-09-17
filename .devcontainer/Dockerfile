FROM mcr.microsoft.com/devcontainers/python:3.12

COPY server/requirements.txt /tmp/server-requirements.txt
COPY ui/requirements.txt /tmp/ui-requirements.txt
RUN pip install --no-cache-dir -r /tmp/server-requirements.txt -r /tmp/ui-requirements.txt
