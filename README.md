# Công cụ chấm điểm tờ bài thi trắc nghiệm THPT Quốc gia 2025 từ ảnh chụp/scan
Link google colab: https://colab.research.google.com/drive/1HaODsV6WNylCiUQbJC-bs2xJ79X5lRZZ?usp=sharing

Đây là dự án cá nhân dùng để rèn luyện các phương pháp xử lý ảnh số, với mục tiêu là scan kết quả từ tờ bài làm trắc nghiệm THPTQG 2025. Hướng tiếp cận này không sử dụng máy đọc điểm quang học OMR mà chỉ sử dụng hình ảnh chụp/scan phiếu trả lời trắc nghiệm.

Dự án này sử dụng các kỹ thuật xử lý ảnh với OpenCV để:

- **Nhận dạng thông tin thí sinh:** Trích xuất số báo danh (SBD) và mã đề thi từ ảnh chụp bài làm hoặc scan từ máy in.
- **Chấm điểm tự động:** Đọc và chấm điểm các phần trắc nghiệm (Phần 1, Phần 2, Phần 3) dựa trên các ô được tô (cần nhập đáp án đề theo đúng format).
- **Xử lý các dạng câu hỏi khác nhau:** Hỗ trợ các dạng câu hỏi trắc nghiệm nhiều lựa chọn, đúng/sai và điền số trong đề thi THPTQG môn Toán.

**Tính năng chính:**

- Sử dụng OpenCV và các thư viện liên quan (`numpy`, `imutils`).
- Áp dụng các bước tiền xử lý ảnh như perspective transformation, grayscale, thresholding.
- Chia tờ bài làm thành các phần: sbd, mã đề, phần 1, phần 2, phần 3.
- Xác định các ô được tô dựa trên số lượng pixel được tô.
- Trích xuất và định dạng kết quả cho từng phần của bài thi.

**Yêu cầu:**

- Python 3.x
- OpenCV
- numpy
- imutils

**Cách sử dụng:**

- Sử dụng local storage của google colab hoặc đăng ảnh lên các công cụ host ảnh free, như https://freeimage.host/.
- Ảnh demo: https://iili.io/FPWy7sf.jpg ; Tờ đề bài chưa tô: https://iili.io/FPXy7Gj.jpg
- Nếu tải về và chạy dưới dạng file python, cần cài các thư viện yêu cầu bên trên và chuyển cv2_imshow thành cv2.imshow
- EXAM_CODE_NUMBER: số mã đề thi, tờ đề 2025 có 2 loại, 1 loại 3 số và 1 loại 4 số
- CUSTOM_THRESHOLD: threshold custom nếu phương pháp Otsu không thực hiện được/bị lỗi
- Nếu không thể detect được contours do ảnh bị mờ/tương phản cao, sử dụng hàm preprocess_img_threshold và chọn threshold thay vì preprocess_img
- DEMO_CONTOUR_DETECTION: hiển thị các contour detect được và các ô khoanh tròn theo từng bước. Set thành False để tăng tốc độ

**Lưu ý:**

- Ở đây sử dụng findContours thay vì HoughCircles vì tốc độ xử lý nhanh hơn, và có thể xử lý được một số trường hợp tô lem ra ngoài.
- Việc xử lý các vấn đề về chất lượng ảnh có thể cần các cải tiến bổ sung.

**Cải tiến:**

- Với ảnh chụp hoặc ảnh scan không theo chuẩn A4, sử dụng registration marks để lấy tọa độ các vùng tô trắc nghiệm thay vì hard code

**Đóng góp:**

Mọi đóng góp để cải thiện công cụ này đều được hoan nghênh!

**Thông tin liên lạc:**

Nếu cần hỗ trợ hoặc có câu hỏi, có thể tạo issue hoặc liên lạc qua các kênh thông tin dưới đây:

- Email: dhdongphuong27@gmail.com
- SĐT: 0522052709
- Facebook: https://www.facebook.com/dhdongphuong27/

**Kết quả:**

<img width="4960" height="3505" alt="image" src="https://github.com/user-attachments/assets/cdfcad92-643e-4535-a7e6-b5e66f4cf784" />

