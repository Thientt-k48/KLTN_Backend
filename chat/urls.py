from django.urls import path
from .views import ChatAPIView, CreateChatSessionView, ChatPredictView, SaveChatLogView, GetChatHistoryView

urlpatterns = [
    # Đường dẫn: /api/chat/session
    path('session', CreateChatSessionView.as_view(), name='create_session'),
    path('predict', ChatPredictView.as_view(), name='chat_predict'),
    path('log', SaveChatLogView.as_view(), name='save_chat_log'),
    path('history', GetChatHistoryView.as_view(), name='chat_history'),
    path('', ChatAPIView.as_view(), name='api_chat'),
]