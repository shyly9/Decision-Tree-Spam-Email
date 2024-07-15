from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import re
from C45 import DecisionTreeC45

app = Flask(__name__)

# Đọc mô hình đã huấn luyện từ tệp
loaded_model = joblib.load('decision_tree_model.joblib')
    
# Thư mục lưu trữ spam và non-spam
SPAM_FOLDER = 'spam'
NON_SPAM_FOLDER = 'non_spam'

# Tạo thư mục 'spam' và 'non_spam' nếu chúng chưa tồn tại
if not os.path.exists(SPAM_FOLDER):
    os.makedirs(SPAM_FOLDER)

if not os.path.exists(NON_SPAM_FOLDER):
    os.makedirs(NON_SPAM_FOLDER)


@app.route('/')
def index():
    return render_template('index.html')

#Xử lý phần nhập email
@app.route('/text', methods=['POST'])
def text():
    if request.method == 'POST':
        # Lấy dữ liệu từ form
        email_text = request.form['email']

        # Xử lý đặc trưng
        features = extract_features(email_text)

        # Chuyển đổi đặc trưng và tên đặc trưng thành DataFrame
        data = pd.DataFrame([features])
        X_predict = data.iloc[:].values

        # Dự đoán kết quả bằng mô hình
        result = loaded_model.predict(X_predict)
        if result == 0:
            result = 'Email này không phải spam'
        else:
            result = 'Email này là spam'
        return render_template('index.html', result=result)

#Xử lý phần upload file
@app.route('/file', methods=['GET', 'POST'])
def file():
    if request.method == 'POST':
        # Lấy tệp tin được upload
        files = request.files.getlist('files[]')
        
        # Duyệt qua từng file
        uploads = []
        for file in files:
            try:
                base_name, _ = os.path.splitext(secure_filename(file.filename))
                # Đọc nội dung từ file sử dụng 'latin-1' encoding
                file_content = file.read().decode('latin-1')

                # Xử lý đặc trưng
                features = extract_features(file_content)

                # Chuyển đổi đặc trưng và tên đặc trưng thành DataFrame
                data = pd.DataFrame([features])
                X_predict = data.iloc[:].values

                # Dự đoán kết quả bằng mô hình
                result = loaded_model.predict(X_predict)
                if result == 0:
                    result = 'Không phải spam'
                else:
                    result = 'Là spam'

                # Lưu thông tin vào danh sách uploads
                uploads.append({
                    'filename': base_name,
                    'file_content': file_content,
                    'result': result
                })

                # Phân loại vào thư mục spam hoặc non-spam
                if result == 'Là spam':
                    destination_folder = SPAM_FOLDER
                else:
                    destination_folder = NON_SPAM_FOLDER

                # Di chuyển file vào thư mục tương ứng
                destination_path = os.path.join(destination_folder, f"{base_name}.txt")
                with open(destination_path, 'w', encoding='utf-8') as destination_file:
                    destination_file.write(file_content)

            except UnicodeDecodeError as e:
                # Handle decoding error, e.g., log the error
                print(f"Error decoding file: {e}")

        return render_template('index.html', uploads=uploads)

@app.route('/download_spam/<filename>')
def download_spam(filename):
    return send_from_directory(SPAM_FOLDER, filename)

@app.route('/download_non_spam/<filename>')
def download_non_spam(filename):
    return send_from_directory(NON_SPAM_FOLDER, filename)



# Hàm để trích xuất đặc trưng từ văn bản email
def extract_features(email_text):
    # Tạo một vectorizer để đếm tần suất xuất hiện của các từ
    vectorizer = CountVectorizer(vocabulary=[
        "make", "address", "all", "3d", "our", "over", "remove", "internet",
        "order", "mail", "receive", "will", "people", "report", "addresses",
        "free", "business", "email", "you", "credit", "your", "font", "000",
        "money", "hp", "hpl", "george", "650", "lab", "labs", "telnet", "857",
        "data", "415", "85", "technology", "1999", "parts", "pm", "direct", "cs",
        "meeting", "original", "project", "re", "edu", "table", "conference", ";",
        "(", "[", "!", "$", "#"
    ])

    # Chuyển đổi văn bản thành vector đặc trưng
    feature_vector = vectorizer.transform([email_text]).toarray()

    # Tổng số từ trong văn bản
    total_words = len(re.findall(r'\b\w+\b', email_text.lower()))

    # Tổng số ký tự trong văn bản
    total_chars = len(email_text)

    # Tính tần suất xuất hiện các từ
    word_frequencies = 100 * feature_vector.flatten()[:48] / total_words

    # Tính tần suất xuất hiện các ký tự đặc biệt
    special_char_frequencies = 100 * feature_vector.flatten()[48:54] / total_chars

    # (Đặc trưng 55) Độ dài trung bình của chuỗi chữ in hoa không gián đoạn
    # (Đặc trưng 56) Độ dài của chuỗi chữ in hoa dài nhất không gián đoạn
    # (Đặc trưng 57) Tổng độ dài của các chuỗi chữ in hoa không gián đoạn
    uppercases = [word for word in email_text.split() if word.isupper()]
    avg_len_uppercase = sum(len(word) for word in uppercases) / len(uppercases) if uppercases else 0
    max_len_uppercase = max(len(word) for word in uppercases) if uppercases else 0
    total_len_uppercase = sum(len(word) for word in uppercases)

    features = list(word_frequencies) + list(special_char_frequencies) + [avg_len_uppercase, max_len_uppercase, total_len_uppercase]
    
    return features


if __name__ == '__main__':
    app.run(debug=False, threaded=True, port=5000)