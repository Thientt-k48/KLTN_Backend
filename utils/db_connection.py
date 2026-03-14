# utils/db_connection.py
from pymongo import MongoClient
from neo4j import GraphDatabase
from django.conf import settings

# --- KHỞI TẠO MONGODB  ---
mongo_client = MongoClient(settings.MONGO_URI)
mongo_db = mongo_client[settings.MONGO_DB_NAME]

# --- KHỞI TẠO NEO4J  ---
neo4j_driver = GraphDatabase.driver(
    settings.NEO4J_URI, 
    auth=settings.NEO4J_AUTH
)