import requests
import json
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

class Dataset():
    def __init__(self, dataPath: str = './sarcasm.json', max_words: int = 10000, max_length: int = 25,
                 train_size: float = 0.8, chunk_size: int = 1024, padding: str = 'post', 
                 truncating: str = 'post', oov_token: str = "<OOV>"):
        """
        Khởi tạo lớp Dataset tùy chỉnh để tải và xử lý dữ liệu văn bản.
        
        Tham số:
        - dataPath (str): Đường dẫn đến tệp dữ liệu (mặc định là './sarcasm.json').
        - max_words (int): Số lượng từ tối đa mà tokenizer sẽ xử lý (mặc định là 10000).
        - max_length (int): Độ dài tối đa của chuỗi sau khi padding (mặc định là 25).
        - train_size (float): Tỷ lệ dữ liệu sử dụng cho việc huấn luyện (mặc định là 0.8).
        - chunk_size (int): Kích thước của từng khối dữ liệu tải về (mặc định là 1024).
        - padding (str): Loại padding ('post' hoặc 'pre', mặc định là 'post').
        - truncating (str): Loại cắt bớt chuỗi ('post' hoặc 'pre', mặc định là 'post').
        - oov_token (str): Token đại diện cho từ không có trong từ điển (mặc định là "<OOV>").
        """
        self.dataPath = dataPath
        self.dataset = []
        self.label_dataset = []
        self.tokenizer = None
        self.max_words = max_words
        self.max_length = max_length
        self.train_size = train_size
        self.chunk_size = chunk_size
        self.padding = padding
        self.truncating = truncating
        self.oov_token = oov_token

    def download(self, url: str):
        """
        Tải xuống dữ liệu từ URL.
        
        Tham số:
        - url (str): Địa chỉ URL chứa dữ liệu cần tải xuống.
        """
        try:
            response = requests.get(url)
            response.raise_for_status()

            with open(self.dataPath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=self.chunk_size):
                    f.write(chunk)

            print("Data downloaded successfully.")
        except requests.exceptions.RequestException as e:
            print(f"Error downloading data: {e}")
            return

    def load_data(self):
        """
        Tải dữ liệu từ tệp JSON và phân tách thành các câu và nhãn.
        """
        try:
            with open(self.dataPath, 'r') as f:
                data = json.load(f)

            for item in data:
                self.dataset.append(item['headline'])
                self.label_dataset.append(item['is_sarcastic'])

            self.dataset = np.array(self.dataset)
            self.label_dataset = np.array(self.label_dataset)

        except FileNotFoundError:
            print("Data file not found.")
            return None

    def split_data(self):
        """
        Phân chia dữ liệu thành bộ huấn luyện và kiểm tra.
        
        Trả về:
        - (tuple): Một tuple chứa dữ liệu huấn luyện và kiểm tra, mỗi cái bao gồm các câu và nhãn.
        """
        size = int(len(self.dataset) * self.train_size)
        train_sentences = self.dataset[:size]
        train_labels = self.label_dataset[:size]
        test_sentences = self.dataset[size:]
        test_labels = self.label_dataset[size:]

        return (train_sentences, train_labels), (test_sentences, test_labels)

    def build_tokenizer(self, train_sentences: list):
        """
        Xây dựng tokenizer từ dữ liệu huấn luyện.
        
        Tham số:
        - train_sentences (list): Danh sách các câu huấn luyện.
        
        Trả về:
        - Tokenizer: Một đối tượng Tokenizer đã được huấn luyện.
        """
        self.tokenizer = Tokenizer(num_words=self.max_words, oov_token=self.oov_token)
        self.tokenizer.fit_on_texts(train_sentences)
        return self.tokenizer

    def tokenize(self, sentences: list):
        """
        Token hóa các câu và áp dụng padding.
        
        Tham số:
        - sentences (list): Danh sách các câu cần token hóa.
        
        Trả về:
        - np.array: Mảng numpy chứa các câu đã được token hóa và padding.
        """
        sequences = self.tokenizer.texts_to_sequences(sentences)
        padded_sequences = pad_sequences(sequences, maxlen=self.max_length,
                                         padding=self.padding, truncating=self.truncating)
        return padded_sequences
