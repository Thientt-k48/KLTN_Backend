import boto3
from botocore.client import Config
from pymongo import MongoClient
from neo4j import GraphDatabase
import google.generativeai as genai
from django.conf import settings
from django.db import connection # Sử dụng kết nối DB có sẵn của Django

# --- KHỞI TẠO CÁC KẾT NỐI TỪ SETTINGS.PY ---

# Gemini
genai.configure(api_key=settings.GOOGLE_API_KEY)
gemini_model = genai.GenerativeModel('gemini-2.5-flash')

# Neo4j
neo4j_driver = GraphDatabase.driver(
    settings.NEO4J_URI, 
    auth=settings.NEO4J_AUTH
)

# MongoDB
mongo_client = MongoClient(settings.MONGO_URI)
mongo_db = mongo_client[settings.MONGO_DB_NAME]

# MinIO (S3)
s3_client = boto3.client(
    's3',
    endpoint_url=f"{'https' if settings.MINIO_STORAGE_USE_HTTPS else 'http'}://{settings.MINIO_STORAGE_ENDPOINT}",
    aws_access_key_id=settings.MINIO_STORAGE_ACCESS_KEY,
    aws_secret_access_key=settings.MINIO_STORAGE_SECRET_KEY,
    config=Config(signature_version='s3v4'),
    region_name='us-east-1'
)

# --- CÁC HÀM TRUY XUẤT DỮ LIỆU ---

def get_minio_link(file_source):
    """Tạo link xem tài liệu từ MinIO"""
    if not file_source: return None
    try:
        return s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': settings.MINIO_STORAGE_BUCKET_NAME, 'Key': file_source},
            ExpiresIn=3600
        )
    except Exception as e:
        print(f"❌ Lỗi MinIO: {e}")
        return None

def get_file_source_from_pg(semantic_id, is_question=True):
    """Lấy file_source từ PostgreSQL sử dụng connection của Django"""
    with connection.cursor() as cursor:
        if is_question:
            # Nếu là Question, join qua bảng content_chunks để lấy file_source
            sql = """
                SELECT c.file_source 
                FROM questions q 
                JOIN content_chunks c ON q.chunk_id = c.id 
                WHERE q.semantic_id = %s
            """
        else:
            # Nếu là Chunk, lấy trực tiếp từ bảng content_chunks
            sql = "SELECT file_source FROM content_chunks WHERE semantic_id = %s"
        
        cursor.execute(sql, [semantic_id])
        row = cursor.fetchone()
        return row[0] if row else None

def search_mongodb_chunks(query_vector, threshold=0.8):
    """Tìm kiếm vector trong MongoDB collection 'chunks'"""
    try:
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "vector_index", # Tên index đã tạo trên MongoDB Atlas
                    "path": "vector",
                    "queryVector": query_vector,
                    "numCandidates": 10,
                    "limit": 1
                }
            },
            {
                "$project": {
                    "semantic_id": 1, 
                    "content": 1,
                    "score": {"$meta": "vectorSearchScore"}
                }
            }
        ]
        results = list(mongo_db.chunks.aggregate(pipeline))
        if results and results[0]['score'] >= threshold:
            return results[0]['semantic_id'], results[0]['content'], results[0]['score']
    except Exception as e:
        print(f"❌ MongoDB Vector Search Error: {e}")
    return None, None, 0.0

# --- LUỒNG XỬ LÝ CHÍNH ---

def generate_response(user_question):
    # 1. Embedding câu hỏi
    embed_res = genai.embed_content(
        model="models/text-embedding-004",
        content=user_question,
        task_type="retrieval_query",
        output_dimensionality=768
    )
    vector = embed_res['embedding']

    # 2. Tìm kiếm Hybrid
    sid = None
    match_type = None
    context_text = ""
    score = 0.0

    # Ưu tiên Neo4j (Question)
    with neo4j_driver.session() as session:
        cypher = """
        CALL db.index.vector.queryNodes('question_embeddings', 1, $vec)
        YIELD node, score WHERE score >= 0.9
        RETURN node.semantic_id AS sid, score
        """
        res = session.run(cypher, vec=vector).single()
        if res:
            sid, match_type, score = res['sid'], 'question', res['score']
            # Lấy thêm text từ MongoDB để đưa vào prompt
            q_doc = mongo_db.questions.find_one({"semantic_id": sid})
            context_text = q_doc.get('answer_text', '') if q_doc else ""

    # Nếu Neo4j không thấy, tìm Chunk ở MongoDB
    if not sid:
        sid, context_text, score = search_mongodb_chunks(vector)
        match_type = 'chunk' if sid else None

    # 3. Tổng hợp kết quả
    if sid:
        file_source = get_file_source_from_pg(sid, is_question=(match_type == 'question'))
        doc_link = get_minio_link(file_source)

        prompt = f"Dựa vào kiến thức: {context_text}, trả lời câu hỏi: {user_question}"
        response = gemini_model.generate_content(prompt)

        return {
            "response": response.text,
            "source": "database",
            "doc_link": doc_link,
            "score": score
        }

    # 4. Fallback Gemini
    response = gemini_model.generate_content(user_question)
    return {"response": response.text, "source": "gemini_knowledge"}