#
# sentiment_sarcasm

Đây là dự án dự đoán câu châm biếm bằng tiếng Anh sử dụng mô hình học sâu (LSTM). Mô hình này được huấn luyện để nhận diện câu châm biếm từ dữ liệu văn bản, hỗ trợ trong việc phân tích cảm xúc và các yếu tố châm biếm trong văn bản.

## Yêu cầu

Trước khi chạy các file, hãy đảm bảo bạn đã cài đặt các thư viện cần thiết. Bạn có thể cài đặt chúng bằng cách chạy:

```bash
pip install -r requirements.txt
```

Các thư viện yêu cầu bao gồm:
- `tensorflow`: Để xây dựng và huấn luyện mô hình LSTM.
- `numpy`: Dùng để xử lý và thao tác với dữ liệu số.
- `matplotlib`: Để trực quan hóa kết quả mô hình.

## Hướng dẫn sử dụng

### Bước 1: Huấn luyện mô hình

Chạy file `train.py` để huấn luyện mô hình học sâu (LSTM) với dữ liệu của bạn. Script này sẽ huấn luyện mô hình và lưu các trọng số của mô hình sau khi huấn luyện.

Để huấn luyện mô hình, chạy lệnh sau trong terminal:

```bash
python train.py
```

Khi chạy file `train.py`, mô hình LSTM sẽ được huấn luyện trên dữ liệu có sẵn và lưu trọng số mô hình sau khi huấn luyện thành công.

**Lưu ý**: Đảm bảo bạn đã chuẩn bị dữ liệu huấn luyện theo đúng định dạng yêu cầu trước khi chạy.

### Bước 2: Dự đoán câu mới

Sau khi mô hình đã được huấn luyện, bạn có thể sử dụng file `predict.py` để dự đoán sự châm biếm trong một câu văn bản mới. Để thực hiện dự đoán, chạy lệnh sau:

```bash
python predict.py
```

Khi chạy file `predict.py`, bạn sẽ được yêu cầu nhập câu văn bản mới mà bạn muốn mô hình phân tích. Mô hình sẽ đưa ra kết quả dự đoán về việc câu đó có chứa châm biếm hay không.

Ví dụ:
- **Câu nhập vào**: "Oh, great! Another flat tire!"
- **Kết quả dự đoán**: Châm biếm (sarcasm).

### Cách chạy ví dụ

1. Đầu tiên, bạn chạy file `train.py` để huấn luyện mô hình với dữ liệu của bạn.
2. Sau khi huấn luyện hoàn tất, bạn có thể sử dụng `predict.py` để nhập câu văn bản mới và nhận dự đoán về châm biếm.

## Liên hệ

Nếu bạn gặp phải vấn đề hoặc có bất kỳ câu hỏi nào về dự án, vui lòng liên hệ với tôi qua email:

[khanh091103@gmail.com]

---

Cảm ơn bạn đã sử dụng dự án này!
```
