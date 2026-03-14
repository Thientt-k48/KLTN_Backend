import json
import time
import traceback
import fitz
import numpy as np
import google.generativeai as genai
from django.db import transaction, connection
from django.conf import settings
from .models import Document
from utils.db_connection import mongo_db, neo4j_driver

genai.configure(api_key=settings.GOOGLE_API_KEY)
json_model = genai.GenerativeModel('gemini-2.5-flash', generation_config={"response_mime_type": "application/json"})
embedding_model = 'models/gemini-embedding-001'



def insert_to_3_databases(grade_track, topic_code, lesson_name, original_id, file_source, chunk_content, chunk_vector, keywords, questions):
    """ 
    Nạp dữ liệu: Postgres (content_chunks, questions), 
    Mongo (chunks có metadata, questions có answer), 
    Neo4j (HAS_GRADE, HAS_TOPIC, HAS_LESSON, HAS_CHUNK, HAS_QUESTION) 
    """
    inserted_mongo_chunk_ids = []
    inserted_mongo_question_ids = []
    
    try:
        with transaction.atomic(): 
            # ==========================================
            # 1. POSTGRESQL (Bảng content_chunks)
            # ==========================================
            with connection.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO content_chunks (file_source, grade_track, lesson_name, original_id) 
                    VALUES (%s, %s, %s, %s) RETURNING id, semantic_id;
                """, [file_source, grade_track, lesson_name, original_id])
                chunk_db_id, semantic_id = cursor.fetchone()

            # ==========================================
            # 2. MONGODB (Collection chunks - Lưu metadata và từ khóa)
            # ==========================================
            mongo_chunk_doc = {
                "semantic_id": semantic_id, 
                "content": chunk_content,
                "keywords": keywords,
                "metadata": {
                    "grade_track": grade_track,
                    "topic_code": topic_code,
                    "lesson_name": lesson_name
                }
            }
            res_chunk = mongo_db.chunks.insert_one(mongo_chunk_doc)
            inserted_mongo_chunk_ids.append(res_chunk.inserted_id)

            # ==========================================
            # 3. NEO4J (Xây dựng Đồ thị Top-Down và Lưu Vector)
            # ==========================================
            with neo4j_driver.session() as session:
                baihoc_id = semantic_id.split('_')[2] if '_' in semantic_id else "bai_chung"
                
                cypher_query = """
                MERGE (th:Thing {id: "TH", name: "Tin học"})
                
                MERGE (l:Lop {id: $grade_track, name: "Lớp " + $grade_track})
                MERGE (th)-[:HAS_GRADE]->(l)
                
                MERGE (cd:ChuDe {id: $topic_code})
                MERGE (l)-[:HAS_TOPIC]->(cd)
                
                MERGE (b:BaiHoc {id: $baihoc_id, name: $lesson_name})
                MERGE (cd)-[:HAS_LESSON]->(b)
                
                MERGE (c:Chunk {semantic_id: $semantic_id})
                SET c.vector = $chunk_vector
                MERGE (b)-[:HAS_CHUNK]->(c)
                """
                session.run(cypher_query, 
                            grade_track=grade_track, topic_code=topic_code, 
                            baihoc_id=baihoc_id, lesson_name=lesson_name,
                            semantic_id=semantic_id, chunk_vector=chunk_vector)

            # ==========================================
            # 4. XỬ LÝ CÂU HỎI (Questions)
            # ==========================================
            if questions:
                # Lưu ý: 'questions' bây giờ phải là mảng các dict: [{"question": "...", "answer": "..."}]
                for q_data in questions:
                    q_text = q_data.get('question', '')
                    q_answer = q_data.get('answer', '')
                    
                    # 4.1. Postgres (Bảng questions)
                    with connection.cursor() as cursor:
                        cursor.execute("""
                            INSERT INTO questions (chunk_id, question_text) 
                            VALUES (%s, %s) RETURNING id, semantic_id;
                        """, [chunk_db_id, q_text])
                        q_semantic_id = cursor.fetchone()[1]

                    # 4.2. Nhúng Vector cho câu hỏi
                    q_vector = genai.embed_content(model=embedding_model, content=q_text)['embedding']
                    
                    # 4.3. Mongo (Collection questions - Lưu thêm answer và tham chiếu chunk)
                    res_q = mongo_db.questions.insert_one({
                        "semantic_id": q_semantic_id,
                        "chunk_semantic_id": semantic_id,
                        "question_text": q_text,
                        "answer": q_answer
                    })
                    inserted_mongo_question_ids.append(res_q.inserted_id)
                    
                    # 4.4. Neo4j (Nút Question và HAS_QUESTION)
                    with neo4j_driver.session() as session:
                        session.run("""
                        MATCH (c:Chunk {semantic_id: $chunk_semantic_id})
                        MERGE (q:Question {semantic_id: $q_semantic_id})
                        SET q.vector = $q_vector
                        MERGE (c)-[:HAS_QUESTION]->(q)
                        """, chunk_semantic_id=semantic_id, q_semantic_id=q_semantic_id, q_vector=q_vector)

        return semantic_id

    except Exception as e:
        if inserted_mongo_chunk_ids:
            mongo_db.chunks.delete_many({"_id": {"$in": inserted_mongo_chunk_ids}})
        if inserted_mongo_question_ids:
            mongo_db.questions.delete_many({"_id": {"$in": inserted_mongo_question_ids}})
        raise Exception(f"Lỗi nạp DB: {e}")


def update_hierarchical_vectors():
    """ 
    Tính toán Vector Trung Bình từ dưới lên (Bottom-up)
    Dựa trên cấu trúc HAS_CHUNK, HAS_LESSON, HAS_TOPIC, HAS_GRADE
    """
    print("[VECTOR MATH] Đang tính toán Vector phân cấp cho mạng lưới Neo4j...")
    
    with neo4j_driver.session() as session:
        # 1. Bài Học (Trung bình của các Chunk con)
        result_baihoc = session.run("MATCH (b:BaiHoc)-[:HAS_CHUNK]->(c:Chunk) RETURN b.id AS id, c.vector AS vec")
        baihoc_dict = {}
        for record in result_baihoc:
            b_id = record["id"]
            if b_id not in baihoc_dict: baihoc_dict[b_id] = []
            if record["vec"]: baihoc_dict[b_id].append(record["vec"])
            
        for b_id, vectors in baihoc_dict.items():
            if vectors:
                avg_vec = np.mean(vectors, axis=0).tolist()
                session.run("MATCH (b:BaiHoc {id: $id}) SET b.vector = $vec", id=b_id, vec=avg_vec)

        # 2. Chủ Đề (Trung bình của các Bài Học con)
        result_chude = session.run("MATCH (cd:ChuDe)-[:HAS_LESSON]->(b:BaiHoc) RETURN cd.id AS id, b.vector AS vec")
        chude_dict = {}
        for record in result_chude:
            cd_id = record["id"]
            if cd_id not in chude_dict: chude_dict[cd_id] = []
            if record["vec"]: chude_dict[cd_id].append(record["vec"])
            
        for cd_id, vectors in chude_dict.items():
            if vectors:
                avg_vec = np.mean(vectors, axis=0).tolist()
                session.run("MATCH (cd:ChuDe {id: $id}) SET cd.vector = $vec", id=cd_id, vec=avg_vec)

        # 3. Lớp (Trung bình của các Chủ Đề con)
        result_lop = session.run("MATCH (l:Lop)-[:HAS_TOPIC]->(cd:ChuDe) RETURN l.id AS id, cd.vector AS vec")
        lop_dict = {}
        for record in result_lop:
            l_id = record["id"]
            if l_id not in lop_dict: lop_dict[l_id] = []
            if record["vec"]: lop_dict[l_id].append(record["vec"])
            
        for l_id, vectors in lop_dict.items():
            if vectors:
                avg_vec = np.mean(vectors, axis=0).tolist()
                session.run("MATCH (l:Lop {id: $id}) SET l.vector = $vec", id=l_id, vec=avg_vec)

        # 4. Thing (Trung bình của các Lớp)
        result_thing = session.run("MATCH (th:Thing)-[:HAS_GRADE]->(l:Lop) RETURN th.id AS id, l.vector AS vec")
        thing_dict = {}
        for record in result_thing:
            th_id = record["id"]
            if th_id not in thing_dict: thing_dict[th_id] = []
            if record["vec"]: thing_dict[th_id].append(record["vec"])
            
        for th_id, vectors in thing_dict.items():
            if vectors:
                avg_vec = np.mean(vectors, axis=0).tolist()
                session.run("MATCH (th:Thing {id: $id}) SET th.vector = $vec", id=th_id, vec=avg_vec)

        print("[VECTOR MATH] Hoàn tất cập nhật cấu trúc Hierarchical Vectors!")

def run_etl_pipeline(doc_id):
    """ 
    Luồng ETL Two-Pass: Đọc nguyên sách lấy cấu trúc -> Xử lý từng bài học -> Nạp DB -> Tính Vector
    """
    uploaded_pdf = None
    try:
        doc = Document.objects.get(id=doc_id)
        file_path = doc.file.path
        grade_track = str(doc.grade)
        
        print(f"[ETL JOB] Bắt đầu xử lý sách: {doc.title} (Lớp {grade_track})")
        doc.status = 'processing'
        doc.save()

        # =================================================================
        # PASS 1: UPLOAD NGUYÊN CUỐN SÁCH & LẤY CẤU TRÚC MỤC LỤC
        # =================================================================
        print("[ETL JOB] Đang tải toàn bộ sách lên Google Cloud...")
        uploaded_pdf = genai.upload_file(path=file_path, display_name=doc.title)
        
        # Chờ file được Google xử lý xong
        while uploaded_pdf.state.name == "PROCESSING":
            print(".", end="")
            time.sleep(2)
            uploaded_pdf = genai.get_file(uploaded_pdf.name)
            
        print("\n[ETL JOB] Đang nhờ AI phân tích cấu trúc vĩ mô của sách...")
        prompt_pass_1 = """
        Bạn là chuyên gia giáo dục. Hãy đọc toàn bộ sách giáo khoa này và lập cấu trúc Mục lục.
        Trả về JSON với mảng "topics" (Chủ đề). Mỗi chủ đề gồm:
        - "topic_code": Mã chủ đề viết liền không dấu (VD: "chude1", "chude_mangmaytinh")
        - "lessons": Mảng các bài học thuộc chủ đề này. Mỗi bài học gồm:
            - "lesson_name": Tên bài học
            - "start_page": Trang bắt đầu của bài học (dựa vào số trang vật lý của file PDF, đếm từ 1)
            - "end_page": Trang kết thúc của bài học
        Tuyệt đối không trích xuất nội dung chi tiết để tránh tràn bộ nhớ.
        """
        response_p1 = json_model.generate_content([uploaded_pdf, prompt_pass_1])
        book_structure = json.loads(response_p1.text)
        print(f"[ETL JOB] Tìm thấy {len(book_structure.get('topics', []))} Chủ đề. Bắt đầu xử lý chi tiết...")

        # =================================================================
        # PASS 2: XỬ LÝ SÂU TỪNG BÀI HỌC DỰA TRÊN TỌA ĐỘ TRANG
        # =================================================================
        pdf_document = fitz.open(file_path)
        
        for topic in book_structure.get('topics', []):
            topic_code = topic.get('topic_code', 'chude_chung')
            
            for lesson in topic.get('lessons', []):
                lesson_name = lesson['lesson_name']
                # Xử lý Index của PyMuPDF (đếm từ 0)
                start_p = max(0, lesson['start_page'] - 1) 
                end_p = min(len(pdf_document) - 1, lesson['end_page'] - 1)
                
                print(f"[ETL JOB] Đang xử lý: [{topic_code}] - {lesson_name} (Trang {start_p+1} đến {end_p+1})")
                
                # Cắt một đoạn PDF nhỏ chỉ chứa bài học hiện tại
                temp_pdf = fitz.open()
                temp_pdf.insert_pdf(pdf_document, from_page=start_p, to_page=end_p)
                temp_pdf_bytes = temp_pdf.write()
                temp_pdf.close()
                
                prompt_pass_2 = f"""
                Đây là nội dung của bài học: "{lesson_name}".
                Hãy đọc kỹ và chia nội dung thành các đoạn (chunk) kiến thức ý nghĩa.
                Trả về JSON có mảng "chunks". Mỗi chunk gồm:
                1. "content": Nội dung chi tiết của đoạn đó.
                2. "keywords": Mảng tối đa 3 từ khóa chuyên ngành.
                3. "questions": Mảng tối đa 2 câu hỏi giả định học sinh có thể hỏi. Phải trả về mảng các Object có dạng: {{"question": "câu hỏi?", "answer": "câu trả lời chi tiết bám sát nội dung chunk"}}
                """
                
                try:
                    # Gửi riêng file byte bài học này cho Gemini xử lý
                    response_p2 = json_model.generate_content([
                        {"mime_type": "application/pdf", "data": temp_pdf_bytes}, 
                        prompt_pass_2
                    ])
                    lesson_data = json.loads(response_p2.text)
                    
                    # Nạp vào 3 DB
                    for chunk_idx, chunk in enumerate(lesson_data.get('chunks', [])):
                        chunk_vector = genai.embed_content(model=embedding_model, content=chunk['content'])['embedding']
                        original_id = f"PAGE{start_p+1}_TO_{end_p+1}_CHUNK{chunk_idx+1}"
                        file_source = f"{doc.file.name}#page={start_p+1}"
                        
                        insert_to_3_databases(
                            grade_track=grade_track,
                            topic_code=topic_code,
                            lesson_name=lesson_name,
                            original_id=original_id,
                            file_source=file_source,
                            chunk_content=chunk['content'],
                            chunk_vector=chunk_vector,
                            keywords=chunk.get('keywords', []),
                            questions=chunk.get('questions', []) # Đã có cả question và answer
                        )
                except Exception as inner_e:
                    print(f"⚠️ Cảnh báo: Lỗi khi xử lý bài {lesson_name}: {inner_e}. Hệ thống sẽ bỏ qua bài này và chạy tiếp.")
                
                # Giảm xóc Rate Limit (Google API)
                time.sleep(5) 

        # =================================================================
        # PASS 3: TÍNH VECTOR PHÂN CẤP & HOÀN TẤT
        # =================================================================
        update_hierarchical_vectors()

        doc.status = 'completed'
        doc.save()
        print(f"[ETL JOB] ✅ THÀNH CÔNG! Đã nạp dữ liệu xong cho: {doc.title}")

    except Exception as e:
        doc.status = 'failed'
        doc.save()
        print(f"[ETL JOB] ❌ THẤT BẠI: Quá trình ETL gặp lỗi!")
        traceback.print_exc()

    finally:
        # LUÔN LUÔN DỌN RÁC CLOUD DÙ CODE CHẠY THÀNH CÔNG HAY BÁO LỖI
        if uploaded_pdf:
            try:
                genai.delete_file(uploaded_pdf.name)
                print("[ETL JOB] Đã dọn dẹp file PDF tạm trên Google Cloud.")
            except Exception as delete_err:
                print(f"[ETL JOB] Không thể xóa file trên Cloud: {delete_err}")