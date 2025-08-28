# import the necessary packages
import numpy as np
import glob
import os
from imutils import contours
import cv2
from scan_answer_sheet import AnswerSheetScanner
from calculate_score import grade_exam


image_paths = glob.glob(os.path.join("input", "*.*")) 
images = []
for path in image_paths:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is not None:  
      images.append(img)
  
scanner = AnswerSheetScanner(SAVE_SCAN_RESULT=True)
for image in images:
  result = scanner.scan_answer(image)

if len(sys.argv) > 1:
    input_text = sys.argv[1]
    grade_exam("output/"+input_text+".csv", "output/*_result.csv")



