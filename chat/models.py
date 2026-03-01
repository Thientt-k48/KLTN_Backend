from django.db import models
from django.contrib.auth.models import User
import uuid
from django.conf import settings

# 1. Bảng lưu phiên chat (Session) - API 14
class ChatSession(models.Model):
    session_id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True, blank=True)
    # THÊM DÒNG NÀY ĐỂ LƯU TIÊU ĐỀ CHAT
    title = models.CharField(max_length=255, default="Cuộc trò chuyện mới")
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.title} ({self.session_id})"

# 2. Bảng lưu nội dung tin nhắn - API 16, 17
class ChatMessage(models.Model):
    ROLE_CHOICES = [('user', 'User'), ('assistant', 'Bot')]
    session = models.ForeignKey(ChatSession, on_delete=models.CASCADE, related_name='messages')
    role = models.CharField(max_length=10, choices=ROLE_CHOICES)
    content = models.TextField()
    sources = models.JSONField(null=True, blank=True) 
    created_at = models.DateTimeField(auto_now_add=True)