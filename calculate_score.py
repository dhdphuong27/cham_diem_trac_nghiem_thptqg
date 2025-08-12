import pandas as pd
import glob
import os
import re

def grade_exam(answer_file, student_files_pattern, output_file="scores.csv", tol=1e-6):
    # Tạo danh sách tên cột (không có student_id, exam_code)
    columns = [f"part1_Q{i}" for i in range(1, 13)]
    for q in range(1, 5):
        for choice in ["A", "B", "C", "D"]:
            columns.append(f"part2_Q{q}_{choice}")
    columns += [f"part3_Q{i}" for i in range(1, 7)]

    # Đọc file đáp án chuẩn
    answer_df = pd.read_csv(answer_file, header=None, names=columns)
    if len(answer_df) != 1:
        raise ValueError("File đáp án chuẩn phải chỉ có 1 dòng.")
    answers = answer_df.iloc[0]

    # Lấy danh sách file bài làm (trừ file đáp án)
    student_files = [
        f for f in glob.glob(student_files_pattern)
        if os.path.abspath(f) != os.path.abspath(answer_file) and os.path.getsize(f) > 0
    ]

    results = []

    for file in student_files:
        # Lấy student_id và exam_code từ tên file
        filename = os.path.basename(file)
        match = re.match(r"(\d+)_(\d+)_result\.csv", filename)
        if not match:
            print(f"⚠️ Bỏ qua file {file} vì tên không đúng định dạng.")
            continue
        student_id, exam_code = match.groups()

        # Đọc dữ liệu bài làm
        try:
            df = pd.read_csv(file, header=None, names=columns)
        except pd.errors.EmptyDataError:
            print(f"⚠️ Bỏ qua file rỗng: {file}")
            continue

        # Nếu file chứa nhiều dòng (hiếm gặp) → chấm từng dòng
        for _, row in df.iterrows():
            score = 0.0

            # ==== PHẦN 1 ====
            for i in range(1, 13):
                if row[f"part1_Q{i}"] == answers[f"part1_Q{i}"]:
                    score += 0.25

            # ==== PHẦN 2 ====
            for q in range(1, 5):
                correct_choices = 0
                wrong_choices = 0
                for choice in ["A", "B", "C", "D"]:
                    col = f"part2_Q{q}_{choice}"
                    if row[col] == answers[col]:
                        correct_choices += 1
                    else:
                        if row[col] is True and answers[col] is False:
                            wrong_choices += 1
                if wrong_choices == 0:
                    if correct_choices == 1:
                        score += 0.1
                    elif correct_choices == 2:
                        score += 0.25
                    elif correct_choices == 3:
                        score += 0.5
                    elif correct_choices == 4:
                        score += 1.0

            # ==== PHẦN 3 ====
            for i in range(1, 7):
                col = f"part3_Q{i}"
                try:
                    val_student = float(row[col])
                    val_answer = float(answers[col])
                    if abs(val_student - val_answer) <= tol:
                        score += 0.5
                except:
                    if row[col] == answers[col]:
                        score += 0.5

            results.append({
                "student_id": student_id,
                "exam_code": exam_code,
                "score": round(score, 2)
            })

    # Xuất kết quả
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    print(f"✅ Đã chấm xong, lưu kết quả vào {output_file}")

if __name__ == "__main__":
    grade_exam("dap_an.csv", "output/*_result.csv")
