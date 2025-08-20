# NIDS-using-traditional-machine-learning-classifier 
(An English version is provided later in this file. Scroll down if needed to see it.)
=========================================================
NIÊN LUẬN CƠ SỞ - NGÀNH AN TOÀN THÔNG TIN
Đề tài: Thống kê và đánh giá các mô hình máy học truyền thống trong phát hiện xâm nhập mạng
Sinh viên: Ngô Đức Thắng (MSSV: B2203737)
Giảng viên hướng dẫn: TS. Nguyễn Hữu Vân Long
=========================================================
1. GIỚI THIỆU
Dự án tập trung vào việc so sánh và đánh giá hiệu năng của các mô hình học máy truyền thống
trong phát hiện xâm nhập mạng, bao gồm:
- K-Nearest Neighbors (KNN)
- Gaussian Naive Bayes (GNB)
- Decision Tree (DT)
- Random Forest (RF)
- XGBoost
Tập dữ liệu: CSE-CIC-IDS2018 (phiên bản rút gọn trên Kaggle). Có thể truy cập vào đường link này để tham khảo dataset: https://www.kaggle.com/datasets/dhoogla/csecicids2018
Các bước xử lý dữ liệu: làm sạch, chuẩn hóa, lựa chọn đặc trưng, cân bằng dữ liệu (SMOTE, Random UnderSampler).

2. CẤU TRÚC FILE CODE
Dự án được chia thành nhiều file code, thực hiện theo thứ tự:
0_MergeAndConver.ipynb      # Tiền xử lý ban đầu: chuyển kiểu dữ liệu PARQUET thành CSV và gộp 10 file thành 1 file duy nhất
1_Visualization.ipynb       # Trực quan hóa dữ liệu
2_DataCleaning.ipynb        # Loại bỏ dữ liệu nhiễu, trùng lặp, cột không cần thiết
3_LabelEncoding.ipynb       # Mã hóa nhãn thành 0 (benign) và 1 (attack)
4_Normalization.ipynb       # Chuẩn hóa dữ liệu (StandardScaler)
5_FeatureSelection.ipynb    # Lựa chọn đặc trưng bằng Random Forest
6a_Balancing.ipynb          # Cân bằng dữ liệu bằng SMOTE
6b_Balancing.ipynb          # Cân bằng dữ liệu bằng Random UnderSampler
7_Splitting.ipynb           # Chia dữ liệu thành tập train/test
8_KNNclassifier.ipynb              # Huấn luyện và đánh giá mô hình KNN
9_NaiveBayesclassifier.ipynb       # Huấn luyện và đánh giá mô hình GaussianNB
10_DecisionTreeclassifier.ipynb    # Huấn luyện và đánh giá mô hình Decision Tree
11_RFclassifier.ipynb              # Huấn luyện và đánh giá mô hình Random Forest
12_XGBoostclassifier.ipynb         # Huấn luyện và đánh giá mô hình XGBoost 

3. YÊU CẦU HỆ THỐNG
- Python 3.13.3
- Môi trường thực hiện: Visual Studio Code với Jupyter Notebook extension hoặc IDE Jupyter Notebook.
- Các thư viện:
  pandas
  numpy
  scikit-learn
  matplotlib
  seaborn
  xgboost
  imbalanced-learn (imblearn)
  joblib
Cài đặt nhanh bằng:
pip install -r requirements.txt

4. CÁCH THỰC THI
- Mở các file code theo thứ tự 00 → 11 bằng Jupyter Notebook hoặc Visual Studio Code có Jupyter Notebook extensions.
- Chạy toàn bộ cell trong từng notebook trước khi chuyển sang file tiếp theo, cụ thể hơn
  + Nếu dùng 1 dataset khác và dữ liệu đưa vào không phải kiểu PARQUET, hãy skip file 0_MergeAndConver.ipynb và vui lòng tự thiết kế một file code khác để chuyển kiểu dữ liệu hiện hành của dataset thành kiểu CSV.
  + Các file code từ 1 đến 5 thực hiện từng cell một như thường, sau mỗi file code sẽ cho ra 1 file kết quả làm đầu vào cho file code tiếp theo cho đến hết. Sau bước này sẽ có 1 tập dữ liệu sạch, chất lượng hơn để huấn luyện nhưng chưa xử lý mất cân bằng.
    . Bước 1 trực quan hóa dữ liệu về số lượng mẫu, đặc trưng, nhãn, mẫu mỗi nhãn.
    . Bước 2 thực hiện làm sạch: xóa bỏ các cột có ít hơn 2% lượng giá trị khác 0, xóa dữ liệu NaN, bị trùng hoặc giá trị vô cùng.
    . Bước 3 thực hiện mã hóa nhãn để phân loại nhị phân: 0 (benign) và 1 (attack).
    . Bước 4 thực hiện chuẩn hóa StandardScaler.
    . Bước 5 thực hiện lựa chọn đặc trưng bằng RandomForest. Có thể điều chỉnh ngưỡng trong dòng "rf15_features = importance_df[importance_df["Importance"] >= 0.025]["Feature"].tolist()", cụ thể là thay 0.025 bằng số khác để thay đổi số lượng đặc trưng cần giữ. Nên đổi tên file kết quả nhận được để tên có nghĩa hơn và không gây nhầm lẫn.
  + Sau khi có được tập dữ liệu ở bước 5, có thể lựa chọn file code ở bước 6a hoặc 6b (hoặc cả hai) để thực hiện xử lý mất cân bằng theo SMOTE hoặc RandomUnderSampler, nhưng phải sửa lại đầu vào ở đầu file bước 6 để nhận được đúng đầu vào là kết quả sau khi thực hiện file code bước 5.
  + Sau xử lý thì chia dữ liệu ở bước thứ 7, bước này sẽ nhận vào tập dữ liệu ở cuối bước 6, thực hiện chia dữ liệu theo kiểu hold-out (80% train và 20% test). Kết quả sẽ là 2 file với tên tập dữ liệu đầu vào gắn thêm 1 trong 2 hậu tố "_train" hoặc "_test" và lưu vào thư mục R2G_output (nếu chưa có code sẽ tự tạo mới, các file code huấn luyện và đánh giá mô hình sẽ nhận đầu vào là các file có đuôi train và test trong thư mục này).
  + Cuối cùng thực hiện các file code từ 8-12 để huấn luyện các mô hình học máy tương ứng KNN, GaussianNB, DecisionTree, RandomForest và XGBoost. Với mỗi file, cần sửa đầu vào để nhận đúng file có đuôi train ở cell code thứ 2, và nhận vào đúng file đuôi test ở cell code thứ 3. Kết quả huấn luyện và đánh giá sẽ được xuất ra trong một file .XLSX, ngoài ra còn có các tệp .JOBLIB để hỗ trợ làm ứng dụng, biểu đồ cột và các ma trận nhầm lẫn (nên chỉnh sửa tên các biểu đồ cho phù hợp vì file được cung cấp lên github sẽ có tên các biểu đồ mặc định là của "undersampler"). Các tổ hợp tham số đã được cài đặt sẵn có dùng tham số mặc định và có tham khảo các nghiên cứu khác. (Các nghiên cứu được tham khảo nằm ở cuối file Word được đính kèm cùng mã nguồn).
- Kết luận, sau khi chạy xong toàn bộ pipeline, ta sẽ có:
  + Các file CSV dữ liệu đã xử lý.
  + File mô hình huấn luyện (.JOBLIB)
  + Kết quả thống kê lưu dưới dạng Excel (.XLSX)
  + Các biểu đồ trực quan (biểu đồ so sánh, ma trận nhầm lẫn)
- Lưu ý: Các file được tạo ra bởi AI bằng các câu hướng dẫn và lệnh của sinh viên thực hiện.

5. KẾT QUẢ
- XGBoost: hiệu năng cao nhất, mô hình nhẹ và nhanh.
- Random Forest & Decision Tree: ổn định, chính xác cao, tuy nhiên RandomForest hơi mất thời gian.
- KNN: hoạt động tốt nhưng tốn bộ nhớ và thời gian.
- GaussianNB: tốc độ nhanh, nhưng precision thấp, không phù hợp IDS.

6. HƯỚNG PHÁT TRIỂN
- Ứng dụng Deep Learning (CNN, LSTM) để cải thiện kết quả.
- Xây dựng pipeline tự động thay vì chạy nhiều file code và phải chỉnh sửa liên tục ở từng file.
- Thử nghiệm thêm trên các tập dữ liệu khác (NSL-KDD, CIC-IDS2017).
- Kết hợp kỹ thuật cân bằng dữ liệu khác (ADASYN).
=========================================================
===============ENGLISH=VERSION===========================
UNDERGRADUATE THESIS – INFORMATION SECURITY
Topic: Statistical Analysis and Evaluation of Traditional Machine Learning Models for Network Intrusion Detection
Student: Ngo Duc Thang (ID: B2203737)
Supervisor: Dr. Nguyen Huu Van Long
=========================================================

1. INTRODUCTION
This project focuses on comparing and evaluating the performance of several traditional machine learning models in the task of network intrusion detection, including:
- K-Nearest Neighbors (KNN)
- Gaussian Naive Bayes (GNB)
- Decision Tree (DT)
- Random Forest (RF)
- XGBoost

Dataset: CSE-CIC-IDS2018 (lightweight version on Kaggle).  
Dataset reference: https://www.kaggle.com/datasets/dhoogla/csecicids2018  
Data preprocessing steps: cleaning, normalization, feature selection, and class imbalance handling (SMOTE, Random UnderSampler).

2. CODE STRUCTURE
The project is organized into multiple Jupyter notebooks, executed sequentially:
0_MergeAndConver.ipynb      # Initial preprocessing: convert PARQUET files to CSV and merge 10 files into one
1_Visualization.ipynb       # Data visualization
2_DataCleaning.ipynb        # Remove noisy data, duplicates, irrelevant features
3_LabelEncoding.ipynb       # Encode labels into binary: 0 (benign), 1 (attack)
4_Normalization.ipynb       # Data normalization (StandardScaler)
5_FeatureSelection.ipynb    # Feature selection using Random Forest
6a_Balancing.ipynb          # Handle imbalance using SMOTE
6b_Balancing.ipynb          # Handle imbalance using Random UnderSampler
7_Splitting.ipynb           # Train-test split (80%/20%)
8_KNNclassifier.ipynb              # Train & evaluate KNN
9_NaiveBayesclassifier.ipynb       # Train & evaluate GaussianNB
10_DecisionTreeclassifier.ipynb    # Train & evaluate Decision Tree
11_RFclassifier.ipynb              # Train & evaluate Random Forest
12_XGBoostclassifier.ipynb         # Train & evaluate XGBoost

3. SYSTEM REQUIREMENTS
- Python 3.13.3
- Development environment: Visual Studio Code with Jupyter Notebook extension, or Jupyter Notebook IDE
- Libraries:
  pandas
  numpy
  scikit-learn
  matplotlib
  seaborn
  xgboost
  imbalanced-learn (imblearn)
  joblib
Quick installation:
pip install -r requirements.txt

4. EXECUTION GUIDE
- Open the notebooks in sequential order (00 → 12) using Jupyter Notebook or VS Code with the Jupyter extension.  
- Run all cells in each notebook before moving to the next one.  
Detailed instructions:
+ If using a different dataset that is not in PARQUET format, skip "0_MergeAndConver.ipynb" and prepare a custom script to convert the dataset into CSV.  
+ Files 1–5 perform stepwise preprocessing. Each notebook outputs a file that serves as the input for the next step. After this stage, you obtain a cleaned dataset ready for training but still imbalanced.  
  • Step 1: Visualize dataset distribution (samples, features, labels).  
  • Step 2: Data cleaning: remove features with <2% non-zero values, drop NaN, duplicates, infinite values. 
  • Step 3: Encode labels into binary: 0 (benign), 1 (attack).  
  • Step 4: Normalize features using StandardScaler.  
  • Step 5: Feature selection via RandomForest. The threshold in the line  
    `"rf15_features = importance_df[importance_df["Importance"] >= 0.025]["Feature"].tolist()"`  
    may be adjusted (e.g., replace 0.025 with another value) to select different numbers of features.  
+ Step 6a or 6b: Choose either SMOTE (oversampling) or RandomUnderSampler (undersampling) for imbalance handling. Ensure the correct input file is specified at the beginning of the notebook.  
+ Step 7: Perform hold-out split (80% train, 20% test). The output consists of two datasets, suffixed with `_train` and `_test`, stored in the "R2G_output" directory.  
+ Steps 8–12: Train and evaluate machine learning models (KNN, GNB, DT, RF, XGBoost). For each model, specify the correct train file in cell 2 and the correct test file in cell 3.  
  - Outputs include: performance results in `.XLSX`, trained models in `.JOBLIB`, bar charts, and confusion matrices. Note: charts are named by default as "undersampler", which should be renamed accordingly to your method.  
  - Parameter settings include both default scikit-learn values and those referenced from prior research (which can be found at the end of the .DOC file included along with the Source code). 
Final outputs of the pipeline:
  + Preprocessed CSV datasets  
  + Trained model files (`.JOBLIB`)  
  + Evaluation reports (`.XLSX`)  
  + Visualization outputs (charts, confusion matrices)  
Note: These files were generated with AI assistance based on the student’s commands and code modifications.

5. RESULTS
- XGBoost: Best overall performance; compact and efficient.  
- Random Forest & Decision Tree: Stable and accurate, though RF is more computationally expensive.  
- KNN: Performs well but memory- and time-intensive.  
- GaussianNB: Very fast, but low precision → unsuitable for IDS.  

6. FUTURE WORK
- Apply Deep Learning models (CNN, LSTM) for improved detection performance.  
- Build an automated pipeline instead of multiple notebooks requiring manual adjustments.  
- Experiment with additional datasets (NSL-KDD, CIC-IDS2017).  
- Explore alternative imbalance-handling techniques (e.g., ADASYN).  
=========================================================
