from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import torch
from kobert_transformers import get_kobert_model, get_tokenizer
import torch.nn as nn

app = FastAPI()

# 감정 라벨 맵핑 (한글 → 영어)
reverse_emotion_label_mapping = {
    0: "neutral",   # 중립
    1: "sadness",   # 슬픔
    2: "anger",     # 분노
    3: "anxiety",   # 불안
    4: "happy",     # 행복
    5: "surprise",  # 당황
    6: "disgust"    # 혐오
}

# 요청 본문 모델 정의
class SentimentRequest(BaseModel):
    text: str

# KoBERT 기반 감정 분석 모델 정의
class SentimentClassifier(nn.Module):
    def __init__(self, bert_model, num_labels):
        super(SentimentClassifier, self).__init__()
        self.bert = bert_model
        for param in self.bert.parameters():
            param.requires_grad = True
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        if hasattr(outputs, "pooler_output"):
            pooled_output = outputs.pooler_output
        else:
            pooled_output = outputs[1]  # Monologg KoBERT
        output = self.drop(pooled_output)
        return self.out(output)

# SentimentAnalyzer 클래스 정의
class SentimentAnalyzer:
    def __init__(self):
        self.tokenizer = get_tokenizer()
        self.model = None

    async def load_model(self):
        if self.model is not None:
            return
        try:
            bert_model = get_kobert_model()
            self.model = SentimentClassifier(bert_model=bert_model, num_labels=7)
            self.model.load_state_dict(torch.load("kobert_sentiment_model.pth", map_location=torch.device('cpu')))
            self.model.eval()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"모델 로드 실패: {str(e)}")

    async def predict(self, text):
        if self.model is None:
            await self.load_model()
        try:
            inputs = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                return_tensors="pt",
                padding='max_length',
                max_length=64,
                truncation=True
            )
            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']

            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs
            predicted_class = torch.argmax(logits, dim=1).item()
            return predicted_class
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"예측 실패: {str(e)}")

# 모델 인스턴스 생성
analyzer = SentimentAnalyzer()

@app.on_event("startup")
async def load_model_on_startup():
    await analyzer.load_model()

# 예측 API 엔드포인트
@app.post("/predict/")
async def predict_sentiment(request: SentimentRequest):  # 본문으로 데이터 받음
    text = request.text  # SentimentRequest에서 텍스트 추출
    predicted_label = await analyzer.predict(text)
    predicted_emotion = reverse_emotion_label_mapping.get(predicted_label, "unknown")
    return {
        "label": predicted_label,
        "emotion": predicted_emotion  # 영어로 반환
    }
