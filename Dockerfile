FROM python:3.11
ADD . .
RUN pip install requests aiohttp python-dotenv aidial-client psycopg2-binary
CMD ["python", "./run.py"] 