import tensorflow as tf
from model.layers.lstm_cell import LSTM

@tf.keras.utils.register_keras_serializable(package="Custom") 
class LSTMModel(tf.keras.Model):
    def __init__(self, units: int, vocab_size: int, embedding_dim: int, input_length: int, **kwargs):
        """
        Khởi tạo mô hình LSTM tùy chỉnh.
        
        Tham số:
        - units (int): Số lượng đơn vị ẩn trong tầng LSTM.
        - vocab_size (int): Kích thước từ vựng.
        - embedding_dim (int): Kích thước vector nhúng của từ.
        - input_length (int): Độ dài tối đa của câu đầu vào.
        """
        super(LSTMModel, self).__init__(**kwargs)
        self.units = units
        self.vocab_size = vocab_size
        self.input_length = input_length
        self.embedding_dim = embedding_dim
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)  # ✅ Xóa input_length
        self.lstm = LSTM(units, inp_shape=embedding_dim)  # ✅ Sửa truyền đối số
        self.lstm.build(None)  # ✅ Khởi tạo weights của LSTM
        self.classfication_model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(64, input_shape=(units,)),  # ✅ Đảm bảo input_shape đúng
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

    def call(self, sentences: tf.Tensor) -> tf.Tensor:
        """
        Truyền dữ liệu qua mô hình.
        
        Tham số:
        - sentences (tf.Tensor): Tensor chứa các câu đầu vào, kích thước (batch_size, input_length).
        
        Trả về:
        - Tensor dự đoán kết quả, kích thước (batch_size, 1).
        """
        batch_size = tf.shape(sentences)[0]

        # Khởi tạo (hidden_state và context_state)
        pre_layer = tf.stack([
            tf.zeros([batch_size, self.lstm.units]),
            tf.zeros([batch_size, self.lstm.units])
        ])

        # Đưa câu qua Embedding để lấy các vector
        embedded_sentences = self.embedding(sentences)

        # Đưa từng từ qua LSTM
        for i in range(self.input_length):
            word = embedded_sentences[:, i, :]
            pre_layer = self.lstm(pre_layer, word)

        h, _ = tf.unstack(pre_layer)
        
        # Sử dụng hidden_state cuối cùng để phân loại
        return self.classfication_model(h)
      
    def get_config(self) -> dict:
        """
        Lưu cấu hình của mô hình để tái sử dụng sau này.
        
        Trả về:
        - dict: Cấu hình của mô hình bao gồm số units, kích thước từ vựng, kích thước nhúng và độ dài đầu vào.
        """
        config = super(LSTMModel, self).get_config()
        config.update({
            "units": self.units,
            "vocab_size": self.vocab_size,
            "embedding_dim": self.embedding_dim,
            "input_length": self.input_length,
        })
        return config

    @classmethod
    def from_config(cls, config: dict):
        """
        Tải lại mô hình từ file cấu hình.
        
        Tham số:
        - config (dict): Cấu hình đã lưu của mô hình.
        
        Trả về:
        - LSTMModel: Mô hình đã được khôi phục.
        """
        return cls(**config)
