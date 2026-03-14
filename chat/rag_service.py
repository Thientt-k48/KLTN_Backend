# chat/rag_service.py
import boto3
from botocore.client import Config
import google.generativeai as genai
from django.conf import settings
from django.db import connection

# Import các trạm trung chuyển DB dùng chung
from utils.db_connection import mongo_db, neo4j_driver

# --- CẤU HÌNH AI & MINIO ---
genai.configure(api_key=settings.GOOGLE_API_KEY)
gemini_model = genai.GenerativeModel('gemini-2.5-flash')
embedding_model = 'models/gemini-embedding-001'

s3_client = boto3.client(
    's3',
    endpoint_url=f"{'https' if settings.MINIO_STORAGE_USE_HTTPS else 'http'}://{settings.MINIO_STORAGE_ENDPOINT}",
    aws_access_key_id=settings.MINIO_STORAGE_ACCESS_KEY,
    aws_secret_access_key=settings.MINIO_STORAGE_SECRET_KEY,
    config=Config(signature_version='s3v4'),
    region_name='us-east-1'
)

# --- CÁC HÀM HỖ TRỢ ---

def get_minio_link(file_source):
    """Tạo presigned URL từ file_source (VD: sach_tin_10.pdf#page=5)"""
    if not file_source: return None
    try:
        parts = file_source.split('#')
        object_name = parts[0]
        fragment = f"#{parts[1]}" if len(parts) > 1 else ""
        
        url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': settings.MINIO_STORAGE_BUCKET_NAME, 'Key': object_name},
            ExpiresIn=3600
        )
        return url + fragment
    except Exception as e:
        print(f"❌ Lỗi tạo link MinIO: {e}")
        return None

def get_file_source_from_pg(chunk_semantic_id):
    """Luôn dùng chunk_semantic_id để lấy file_source từ bảng content_chunks"""
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT file_source FROM content_chunks WHERE semantic_id = %s", [chunk_semantic_id])
            row = cursor.fetchone()
            return row[0] if row else None
    except Exception as e:
        print(f"❌ Lỗi truy vấn PostgreSQL: {e}")
        return None

# --- THUẬT TOÁN TÌM KIẾM TRÊN NEO4J ĐỒ THỊ ---

def search_neo4j_questions(query_vector, threshold=0.85):
    """ Tìm Câu Hỏi giống nhất (Vector Search cơ bản) """
    cypher = """
    MATCH (q:Question)
    // Tính Cosine Similarity (Vector đã chuẩn hóa nên Dot Product = Cosine)
    WITH q, reduce(dot=0.0, i IN range(0, size(q.vector)-1) | dot + q.vector[i] * $query_vector[i]) AS score
    WHERE score >= $threshold
    ORDER BY score DESC LIMIT 1
    
    // Tìm ngược về Chunk chứa câu hỏi này
    MATCH (c:Chunk)-[:HAS_QUESTION]->(q)
    RETURN q.semantic_id AS q_sid, c.semantic_id AS chunk_sid, score
    """
    with neo4j_driver.session() as session:
        result = session.run(cypher, query_vector=query_vector).data()
        if result:
            return result[0]['q_sid'], result[0]['chunk_sid'], result[0]['score']
    return None, None, 0.0

def search_neo4j_hierarchical_chunks(query_vector):
    """ 
    Chiến lược VÉT CẠN TỪ TRÊN XUỐNG (Top-down Routing)
    Lọc Lớp -> Lọc Chủ Đề -> Lọc Bài Học -> Lọc Chunk
    """
    cypher = """
    // Tầng 1: Chọn Lớp giống nhất
    MATCH (l:Lop)
    WITH l, reduce(dot=0.0, i IN range(0, size(l.vector)-1) | dot + l.vector[i] * $query_vector[i]) AS score_l
    ORDER BY score_l DESC LIMIT 1
    
    // Tầng 2: Từ Lớp đó, chọn Chủ Đề giống nhất
    MATCH (l)-[:HAS_TOPIC]->(cd:ChuDe)
    WITH cd, reduce(dot=0.0, i IN range(0, size(cd.vector)-1) | dot + cd.vector[i] * $query_vector[i]) AS score_cd
    ORDER BY score_cd DESC LIMIT 1
    
    // Tầng 3: Từ Chủ Đề đó, chọn Bài Học giống nhất
    MATCH (cd)-[:HAS_LESSON]->(b:BaiHoc)
    WITH b, reduce(dot=0.0, i IN range(0, size(b.vector)-1) | dot + b.vector[i] * $query_vector[i]) AS score_b
    ORDER BY score_b DESC LIMIT 1
    
    // Tầng 4: Từ Bài Học đó, chọn Chunk giống nhất
    MATCH (b)-[:HAS_CHUNK]->(c:Chunk)
    WITH c, reduce(dot=0.0, i IN range(0, size(c.vector)-1) | dot + c.vector[i] * $query_vector[i]) AS score_c
    ORDER BY score_c DESC LIMIT 1
    
    RETURN c.semantic_id AS chunk_sid, score_c AS score
    """
    with neo4j_driver.session() as session:
        result = session.run(cypher, query_vector=query_vector).data()
        if result:
            return result[0]['chunk_sid'], result[0]['score']
    return None, 0.0

# --- LUỒNG XỬ LÝ CHÍNH (RAG WORKFLOW) ---

def generate_response(user_question):
    # 1. Nhúng vector câu hỏi
    try:
        embed_res = genai.embed_content(model=embedding_model, content=user_question)
        query_vector = embed_res['embedding']
    except Exception as e:
        return {"response": "Hệ thống AI đang bận, vui lòng thử lại sau.", "source": "error"}

    match_type = None
    target_chunk_id = None
    final_response = ""
    score = 0.0

    # ==========================================
    # ƯU TIÊN 1: Tìm Câu hỏi do LLM sinh ra (Độ chính xác tuyệt đối)
    # ==========================================
    q_sid, chunk_sid, q_score = search_neo4j_questions(query_vector, threshold=0.85)
    
    if q_sid:
        match_type = 'question'
        target_chunk_id = chunk_sid
        score = q_score
        
        # Móc ngay CÂU TRẢ LỜI (Answer) từ MongoDB (đã lưu ở bước ETL)
        q_doc = mongo_db.questions.find_one({"semantic_id": q_sid})
        if q_doc and "answer" in q_doc:
            final_response = q_doc["answer"]

    # ==========================================
    # ƯU TIÊN 2: Đi men theo Đồ thị Phân cấp (Top-down) để tìm Chunk
    # ==========================================
    if not match_type:
        c_sid, c_score = search_neo4j_hierarchical_chunks(query_vector)
        
        if c_sid and c_score > 0.65: # Threshold cho chunk có thể để thấp hơn chút
            match_type = 'chunk'
            target_chunk_id = c_sid
            score = c_score
            
            # Lấy NỘI DUNG (Content) từ MongoDB để chém gió
            chunk_doc = mongo_db.chunks.find_one({"semantic_id": c_sid})
            if chunk_doc:
                context_text = chunk_doc.get("content", "")
                
                # Ép LLM trả lời bám sát Sách
                prompt = f"""
                Bạn là giáo viên Tin học. Dựa VÀO ĐÚNG nội dung sách giáo khoa dưới đây:
                "{context_text}"
                
                Hãy trả lời câu hỏi của học sinh: "{user_question}". 
                Trình bày rõ ràng, thân thiện. Tuyệt đối không bịa thêm kiến thức ngoài sách.
                """
                response = gemini_model.generate_content(prompt)
                final_response = response.text

    # ==========================================
    # TỔNG HỢP VÀ TRẢ VỀ FRONTEND
    # ==========================================
    if match_type and target_chunk_id and final_response:
        # Lấy file PDF từ bảng content_chunks
        file_source = get_file_source_from_pg(target_chunk_id)
        doc_link = get_minio_link(file_source)

        return {
            "response": final_response,
            "source": f"database_{match_type}",
            "doc_link": doc_link,
            "score": score
        }

    # ==========================================
    # FALLBACK: Không tìm thấy gì, chém gió bằng kiến thức chung
    # ==========================================
    fallback_prompt = f"Bạn là chatbot giáo dục. Học sinh hỏi: {user_question}. Hãy trả lời ngắn gọn, dễ hiểu."
    response = gemini_model.generate_content(fallback_prompt)
    
    return {
        "response": response.text, 
        "source": "gemini_knowledge",
        "doc_link": None,
        "score": 0.0
    }