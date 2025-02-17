import matplotlib.pyplot as plt
from data import Dataset
from model import LSTMModel
import tensorflow as tf
import pickle
import os

# Tạo thư mục lưu kết quả nếu chưa tồn tại
os.makedirs("result", exist_ok=True)

print("Step 1: Loading data...")

dataset = Dataset()
dataset.download('https://raw.githubusercontent.com/ashwaniYDV/sarcasm-detection-tensorflow/refs/heads/main/sarcasm.json')
dataset.load_data()

(train_sentences, train_labels), (test_sentences, test_labels) = dataset.split_data()

tokenizer = dataset.build_tokenizer(train_sentences)

tokenized_train_sentences = dataset.tokenize(train_sentences)
tokenized_test_sentences = dataset.tokenize(test_sentences)

# Lưu tokenizer trước khi train
tokenizer_path = "result/tokenizer.pkl"
with open(tokenizer_path, "wb") as f:
    pickle.dump(tokenizer, f)

print(f"✅ Tokenizer saved to {tokenizer_path}")

print("Step 2: Training...")
units = 128
embedding_size = 100
vocab_size = len(tokenizer.index_word) + 1
max_length = 25
input_length = max_length

lstm_model = LSTMModel(units, vocab_size, embedding_size, input_length)

lstm_model.compile(
  optimizer=tf.keras.optimizers.Adam(0.001), loss='binary_crossentropy', metrics=['acc']
)

history = lstm_model.fit(tokenized_train_sentences, train_labels, 
                          validation_data=(tokenized_test_sentences, test_labels), 
                          batch_size=32, epochs=5)

print("Step 3: Saving model...")
lstm_model.save("result/lstm_sarcasm_model.keras")
print("Model saved successfully.")

# Vẽ đồ thị kết quả huấn luyện
plt.figure(figsize=(10, 5))
plt.plot(history.history['acc'], label='Train Accuracy')
plt.plot(history.history['val_acc'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training vs Validation Accuracy')
plt.legend()

# Lưu đồ thị vào file ảnh
plot_path = "result/training_plot.png"
plt.savefig(plot_path)
print(f"✅ Training plot saved to {plot_path}")
plt.show()
