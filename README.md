# Công cụ chấm điểm bài thi THPT Quốc gia 2025

Đây là một phần của công cụ tự động chấm điểm bài thi trắc nghiệm cho Kỳ thi Tốt nghiệp Trung học Phổ thông Quốc gia năm 2025 tại Việt Nam.

Dự án này sử dụng các kỹ thuật xử lý ảnh với OpenCV để:

- **Nhận dạng thông tin thí sinh:** Trích xuất số báo danh (SBD) và mã đề thi từ ảnh chụp bài làm.
- **Chấm điểm tự động:** Đọc và chấm điểm các phần trắc nghiệm (Phần 1, Phần 2, Phần 3) dựa trên các ô được tô (bubbled).
- **Xử lý các dạng câu hỏi khác nhau:** Hỗ trợ các dạng câu hỏi trắc nghiệm nhiều lựa chọn, đúng/sai và điền số.

**Tính năng chính:**

- Sử dụng OpenCV và các thư viện liên quan (`numpy`, `imutils`).
- Áp dụng các bước tiền xử lý ảnh như chuyển sang ảnh xám, làm mờ và phân ngưỡng Otsu.
- Phát hiện và sắp xếp các ô đáp án (contours).
- Xác định các ô được tô sáng (bubbled indices) dựa trên số lượng pixel khác không.
- Trích xuất và định dạng kết quả cho từng phần của bài thi.

**Yêu cầu:**

- Python 3.x
- OpenCV
- numpy
- imutils

**Cách sử dụng:**

(Hướng dẫn chi tiết về cách cài đặt thư viện và chạy mã sẽ được bổ sung sau.)

**Lưu ý:**

- Mã nguồn hiện tại là một phần của hệ thống chấm điểm hoàn chỉnh.
- Các tham số như tọa độ cắt ảnh (cropping coordinates) được điều chỉnh dựa trên cấu trúc phôi giấy thi cụ thể và có thể cần điều chỉnh nếu sử dụng các phôi giấy khác.
- Việc xử lý các trường hợp đặc biệt như tô nhầm, tô nhiều đáp án, hoặc các vấn đề về chất lượng ảnh có thể cần các cải tiến bổ sung.

**Đóng góp:**

Mọi đóng góp để cải thiện công cụ này đều được hoan nghênh!
