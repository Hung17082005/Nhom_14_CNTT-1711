# Hotel Mining Project
##  Giới thiệu
Dự án này thực hiện phân tích và khai phá dữ liệu (Data Mining) trong lĩnh vực khách sạn nhằm tìm ra các insight quan trọng từ dữ liệu thực tế.

Hệ thống bao gồm các bước:
- Phân tích dữ liệu (EDA)
- Tiền xử lý dữ liệu (Preprocessing)
- Khai phá dữ liệu (Mining)
- Xây dựng mô hình (Modeling)
- Đánh giá (Evaluation)
- Trực quan hóa và triển khai với Streamlit

---

##  Công nghệ sử dụng
- Python 
- Pandas, NumPy
- Scikit-learn
- Matplotlib / Seaborn
- Streamlit (web app)


## Cấu trúc thư mục

hotel_mining_project/
│── configs/ # Cấu hình
│── data/ # Dữ liệu
│── notebooks/ # Notebook phân tích
│── scripts/ # Pipeline chạy chính
│── src/
│ ├── data/ # Load & clean dữ liệu
│ ├── features/ # Feature engineering
│ ├── mining/ # Clustering / Association
│ ├── models/ # Model ML
│ ├── evaluation/ # Đánh giá
│ └── visualization/ # Vẽ biểu đồ
│── streamlit_app.py # Giao diện web
│── requirements.txt
│── README.md


##  Cài đặt

### 1. Clone repo
git clone https://github.com/Hung17082005/Nhom_14_CNTT-1711.git

cd Nhom_14_CNTT-1711

2. Tạo môi trường ảo
python -m venv venv
venv\Scripts\activate   # Windows

3. Cài thư viện
pip install -r requirements.txt
Chạy project
Chạy pipeline:
python scripts/run_pipeline.py
Chạy web app:
streamlit run streamlit_app.py
