import os
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from data import Dataset
from model import LSTMModel  # ✅ Import lớp custom

# ✅ Load model
model_path = "result/lstm_sarcasm_model.keras"

print(f"✅ Loading model from {model_path} ...")
model = load_model(model_path, custom_objects={"LSTMModel": LSTMModel})
print("🎉 Model loaded successfully!")

# ✅ Load tokenizer
tokenizer_path = "result/tokenizer.pkl"

print(f"✅ Loading tokenizer from {tokenizer_path} ...")
with open(tokenizer_path, "rb") as f:
    tokenizer = pickle.load(f)
print("🎉 Tokenizer loaded successfully!")

# ✅ Tạo Dataset giả để dùng hàm tokenize
dataset = Dataset()
dataset.tokenizer = tokenizer  # Gán tokenizer vào Dataset để tái sử dụng tokenize

# ✅ Hàm dự đoán
def predict_sarcasm(sentence):
    tokenized_sentence = dataset.tokenize([sentence])
    prediction = model.predict(tokenized_sentence)
    
    sarcasm_prob = prediction[0][0]
    if sarcasm_prob >= 0.5:
        print(f"🤔 Câu này có thể là châm biếm! (Xác suất: {sarcasm_prob:.2f})")
    else:
        print(f"✅ Câu này không phải châm biếm. (Xác suất: {sarcasm_prob:.2f})")

# ✅ Nhận input từ bàn phím
while True:
    user_input = input("\nNhập câu để kiểm tra ('exit' để thoát): ")
    if user_input.lower() == 'exit':
        break
    predict_sarcasm(user_input)
