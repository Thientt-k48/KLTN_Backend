import boto3
from botocore.client import Config
from pymongo import MongoClient
from neo4j import GraphDatabase
import google.generativeai as genai
from django.conf import settings
from django.db import connection
import numpy as np
import urllib.parse

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
    if not file_source: return None
    try:
        # 1. Bóc tách chuỗi URL thành các thành phần
        parsed_url = urllib.parse.urlparse(file_source)
        
        # 2. Lấy ra phần đường dẫn (VD: /sach-giao-khoa/SACH...pdf)
        path = parsed_url.path
        
        # 3. Cắt bỏ tên bucket ra khỏi path để lấy chính xác Object Key
        bucket_prefix = f"/{settings.MINIO_STORAGE_BUCKET_NAME}/"
        if path.startswith(bucket_prefix):
            object_key = path.replace(bucket_prefix, "", 1)
        else:
            object_key = path.lstrip("/") # Nếu chuỗi chỉ có tên file

        # 4. Giải mã ký tự (Đổi %20 thành dấu cách, v.v.)
        object_key = urllib.parse.unquote(object_key)

        # 5. Tạo link an toàn từ MinIO
        presigned_url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': settings.MINIO_STORAGE_BUCKET_NAME, 'Key': object_key},
            ExpiresIn=3600
        )

        # 6. Gắn lại số trang PDF vào cuối link mới tạo
        if parsed_url.fragment:
            presigned_url = f"{presigned_url}#{parsed_url.fragment}"

        return presigned_url
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

def cosine_similarity(vec1, vec2):
    """Tính độ tương đồng giữa 2 vector"""
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def search_mongodb_chunks(query_vector, threshold=0.8):
    """Tìm kiếm bằng Python thay vì dựa vào MongoDB $vectorSearch"""
    try:
        best_match = None
        highest_score = 0.0
        
        # Lấy tất cả chunks từ MongoDB (Chỉ phù hợp nếu DB nhỏ/vừa)
        all_chunks = mongo_db.chunks.find({}, {"semantic_id": 1, "content": 1, "vector": 1})
        
        for chunk in all_chunks:
            chunk_vector = chunk.get("vector")
            if chunk_vector:
                score = cosine_similarity(query_vector, chunk_vector)
                if score > highest_score and score >= threshold:
                    highest_score = score
                    best_match = chunk
                    
        if best_match:
            return best_match['semantic_id'], best_match['content'], highest_score
            
    except Exception as e:
        print(f"❌ Lỗi xử lý Vector Python: {e}")
        
    return None, None, 0.0

# --- LUỒNG XỬ LÝ CHÍNH ---

def generate_response(user_question):
    # 1. Embedding câu hỏi
    embed_res = genai.embed_content(
        model="models/gemini-embedding-001",
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
            # Lấy CÂU TRẢ LỜI CHUẨN từ MongoDB
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

        # XỬ LÝ TRẢ LỜI: Trực tiếp trả về context_text nếu là Question
        if match_type == 'question':
             final_response = context_text 
        else:
             # Nếu là Chunk, dùng Gemini để sinh câu trả lời dựa trên nội dung
             prompt = f"Dựa vào nội dung tài liệu sau: {context_text}, hãy trả lời ngắn gọn, trực tiếp vào trọng tâm câu hỏi: {user_question}. Không cần giải thích dài dòng."
             response = gemini_model.generate_content(prompt)
             final_response = response.text

        return {
            "response": final_response,
            "source": "database",
            "doc_link": doc_link,
            "score": score
        }

    # 4. Fallback Gemini
    response = gemini_model.generate_content(user_question)
    return {"response": response.text, "source": "gemini_knowledge"}