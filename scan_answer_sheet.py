
import cv2
import numpy as np
import imutils
from imutils import contours
import os
import csv
import pandas as pd
class AnswerSheetScanner:
  def __init__(self, SAVE_SCAN_RESULT=False, EXAM_CODE_NUMBER=4, CUSTOM_THRESHOLD=200, OUTPUT_FOLDER_PATH="output/"):
    self.SAVE_SCAN_RESULT = SAVE_SCAN_RESULT
    self.EXAM_CODE_NUMBER = EXAM_CODE_NUMBER
    self.CUSTOM_THRESHOLD = CUSTOM_THRESHOLD
    self.OUTPUT_FOLDER_PATH = OUTPUT_FOLDER_PATH
    self.RESOLUTION_SCALE = 1.0

  def scan_answer(self, image):
    height, width = image.shape[:2]

    self.RESOLUTION_SCALE = height / 3505

    student_id_section, exam_code_section, part1_section, part1_section_horizontal,part2_section, part2_section_horizontal,part3_section, sign_and_decimal_section_horizontal, numbers_section_horizontal = self.segment_image(image)

    student_id = self.scan_student_id(student_id_section)
    exam_code = self.scan_exam_code(exam_code_section)
    part1_answers = self.scan_part1(part1_section_horizontal)
    part2_answers = self.scan_part2(part2_section_horizontal)
    part3_answers = self.scan_part3(sign_and_decimal_section_horizontal, numbers_section_horizontal)

    result = {
      'student_id': student_id,
      'exam_code': exam_code,
      'part1': part1_answers,
      'part2': part2_answers,
      'part3': part3_answers
    }
    if self.SAVE_SCAN_RESULT:
      output_path_img = os.path.join(self.OUTPUT_FOLDER_PATH, f"{student_id}_{exam_code}_scan.jpg")
      self.save_scan_result(image, output_path_img, student_id_section, exam_code_section, part1_section, part2_section, part3_section)
    output_path_csv = os.path.join(self.OUTPUT_FOLDER_PATH, f"{student_id}_{exam_code}_result.csv")
    list_result = self.dic_to_list(result)
    with open(output_path_csv, "w", newline="") as f:
      writer = csv.writer(f)
      writer.writerow(list_result) 

    return result
  def dic_to_list(self, data):
    results = []
    for q_num in sorted(data['part1'].keys()):
        results.append(data['part1'][q_num][0])
    for q_num in sorted(data['part2'].keys()):
        for opt in ['A', 'B', 'C', 'D']:
            results.append(data['part2'][q_num][opt][0])
    # Part 3 answers
    for q_num in sorted(data['part3'].keys()):
        results.append(float(data['part3'][q_num]))
    return results


  def preprocess_img(self,image):
    if image.ndim == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.ndim == 3 and image.shape[2] == 4:
        gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    else:
        gray = image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # apply Otsu's thresholding method to binarize the warped
    # piece of paper
    thresh = cv2.threshold(blurred, 0, 255,
      cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    return thresh

  def preprocess_img_threshold(self,image):
    if image.ndim == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.ndim == 3 and image.shape[2] == 4:
        gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    else:
        gray = image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # use pre-determined threshold to binarize the warped
    # piece of paper
    thresh = cv2.threshold(blurred, self.CUSTOM_THRESHOLD, 255,
      cv2.THRESH_BINARY_INV)[1]
    return thresh

  def non_max_suppression(self,boxes, overlapThresh):
      if len(boxes) == 0:
          return []

      if boxes.dtype.kind == "i":
          boxes = boxes.astype("float")

      pick = []

      x1 = boxes[:, 0]
      y1 = boxes[:, 1]
      x2 = boxes[:, 2]
      y2 = boxes[:, 3]

      area = (x2 - x1 + 1) * (y2 - y1 + 1)
      idxs = np.argsort(y2)

      while len(idxs) > 0:
          last = len(idxs) - 1
          i = idxs[last]
          pick.append(i)

          xx1 = np.maximum(x1[i], x1[idxs[:last]])
          yy1 = np.maximum(y1[i], y1[idxs[:last]])
          xx2 = np.minimum(x2[i], x2[idxs[:last]])
          yy2 = np.minimum(y2[i], y2[idxs[:last]])

          w = np.maximum(0, xx2 - xx1 + 1)
          h = np.maximum(0, yy2 - yy1 + 1)

          overlap = (w * h) / area[idxs[:last]]

          idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

      return boxes[pick].astype("int")

  def find_contours(self,thresh):
    # Find contours in the thresholded image
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    questionCnts = []

    # Create an array to store bounding boxes
    boundingBoxes = []

    # Loop over the contours and store their bounding boxes
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)
        if (w >= 20*self.RESOLUTION_SCALE and h >= 20*self.RESOLUTION_SCALE and ar >= 0.9 and ar <= 1.1):
            boundingBoxes.append([x, y, x + w, y + h])
            questionCnts.append(c)

    # Convert the bounding boxes list to a numpy array
    boundingBoxes = np.array(boundingBoxes)

    # Apply non-maximum suppression
    filteredBoxes = self.non_max_suppression(boundingBoxes, overlapThresh=0.3)

    # Loop over the filtered boxes to keep only non-overlapping contours
    filteredContours = []
    for (startX, startY, endX, endY) in filteredBoxes:
        for c in questionCnts:
            (x, y, w, h) = cv2.boundingRect(c)
            if startX == x and startY == y and endX == x + w and endY == y + h:
                filteredContours.append(c)
                break
    return filteredContours

  def sort_contours(self,filteredContours, n):
    # sort the question contours top-to-bottom, then initialize
    tmp = contours.sort_contours(filteredContours,
      method="top-to-bottom")[0]

    sorted_ctrs = []

    for (q, i) in enumerate(range(len(filteredContours) - n, -1, -n)):
      sorted_ctrs.extend(contours.sort_contours(filteredContours[i:i + n])[0])
    return sorted_ctrs
  #Here i sort top down - left to right

  def get_bubbled_indicies(self,sorted_ctrs,thresh):
    #Now I have sorted_ctrs, which contains all contours in sorted order
    #Now loop through every contours to check if that countour's innter area contains more black or white pixel
    #in order to determine where that box is checked
    bubbled = None
    bubbled_indices = []
    highest_non_bubbled_area = 500*self.RESOLUTION_SCALE*self.RESOLUTION_SCALE  #500 is the default threshold, any printed number will be lower than this threshold
    # Loop over the sorted contours
    for (j, c) in enumerate(sorted_ctrs):
        # Construct a mask that reveals only the current "bubble" for the question
        mask = np.zeros(thresh.shape, dtype="uint8")
        cv2.drawContours(mask, [c], -1, 255, -1)

        # Apply the mask to the thresholded image, then count the number of non-zero pixels in the bubble area
        mask = cv2.bitwise_and(thresh, thresh, mask=mask)
        total = cv2.countNonZero(mask)
        # Append the index of the current bubble if it has non-zero pixels
        if total > highest_non_bubbled_area:
            bubbled_indices.append(j)
    return bubbled_indices

  def segment_image(self,image):
    student_id_section = image[int(350*self.RESOLUTION_SCALE):int(1052*self.RESOLUTION_SCALE), int(1634*self.RESOLUTION_SCALE):int(2043*self.RESOLUTION_SCALE)]
    exam_code_section = image[int(354*self.RESOLUTION_SCALE):int(1056*self.RESOLUTION_SCALE), int(2093*self.RESOLUTION_SCALE):int(2310*self.RESOLUTION_SCALE)]
    part1_section = image[int(1244*self.RESOLUTION_SCALE):int(1887*self.RESOLUTION_SCALE), int(248*self.RESOLUTION_SCALE):int(2254*self.RESOLUTION_SCALE)]
    part2_section = image[int(2074*self.RESOLUTION_SCALE):int(2319*self.RESOLUTION_SCALE), int(248*self.RESOLUTION_SCALE):int(2232*self.RESOLUTION_SCALE)]
    part3_section = image[int(2530*self.RESOLUTION_SCALE):int(3231*self.RESOLUTION_SCALE), int(254*self.RESOLUTION_SCALE):int(2238*self.RESOLUTION_SCALE)]

    #Part 1 handle
    height, width = part1_section.shape[:2]
    column_width = width // 4
    # Crop the image into 4 vertical columns
    cropped_images = [part1_section[int(5*self.RESOLUTION_SCALE):int(625*self.RESOLUTION_SCALE), int(10*self.RESOLUTION_SCALE):int(446*self.RESOLUTION_SCALE)],
                      part1_section[int(5*self.RESOLUTION_SCALE):int(625*self.RESOLUTION_SCALE), int(520*self.RESOLUTION_SCALE):int(956*self.RESOLUTION_SCALE)],
                      part1_section[int(5*self.RESOLUTION_SCALE):int(625*self.RESOLUTION_SCALE), int(1030*self.RESOLUTION_SCALE):int(1466*self.RESOLUTION_SCALE)],
                      part1_section[int(5*self.RESOLUTION_SCALE):int(625*self.RESOLUTION_SCALE), int(1540*self.RESOLUTION_SCALE):int(1976*self.RESOLUTION_SCALE)]]

    part1_section_horizontal = cv2.vconcat(cropped_images)
    #Part 2 handle
    cropped_images = [part2_section[int(0*self.RESOLUTION_SCALE):int(238*self.RESOLUTION_SCALE), int(62*self.RESOLUTION_SCALE):int(253*self.RESOLUTION_SCALE)],part2_section[int(0*self.RESOLUTION_SCALE):int(238*self.RESOLUTION_SCALE), int(255*self.RESOLUTION_SCALE):int(446*self.RESOLUTION_SCALE)],
                      part2_section[int(0*self.RESOLUTION_SCALE):int(238*self.RESOLUTION_SCALE), int(565*self.RESOLUTION_SCALE):int(756*self.RESOLUTION_SCALE)],part2_section[int(0*self.RESOLUTION_SCALE):int(238*self.RESOLUTION_SCALE), int(764*self.RESOLUTION_SCALE):int(955*self.RESOLUTION_SCALE)],
                      part2_section[int(0*self.RESOLUTION_SCALE):int(238*self.RESOLUTION_SCALE), int(1079*self.RESOLUTION_SCALE):int(1270*self.RESOLUTION_SCALE)],part2_section[int(0*self.RESOLUTION_SCALE):int(238*self.RESOLUTION_SCALE), int(1272*self.RESOLUTION_SCALE):int(1463*self.RESOLUTION_SCALE)],
                      part2_section[int(0*self.RESOLUTION_SCALE):int(238*self.RESOLUTION_SCALE), int(1590*self.RESOLUTION_SCALE):int(1781*self.RESOLUTION_SCALE)],part2_section[int(0*self.RESOLUTION_SCALE):int(238*self.RESOLUTION_SCALE), int(1786*self.RESOLUTION_SCALE):int(1977*self.RESOLUTION_SCALE)]]

    part2_section_horizontal =  cv2.vconcat(cropped_images)

    #Part 3 handle
    height, width = part3_section.shape[:2]
    column_width = width // 6
    # Crop the image into 6 vertical columns
    cropped_images = []
    for i in range(6):  # Six columns
        x_start = i * column_width
        cropped_image = part3_section[:, x_start:x_start + column_width]
        cropped_images.append(cropped_image)
    sign_and_decimal_section = []
    numbers_section = []
    for i in range(len(cropped_images)):
      sign_and_decimal_section.append(cropped_images[i][int(0*self.RESOLUTION_SCALE):int(120*self.RESOLUTION_SCALE),:])
      numbers_section.append(cropped_images[i][int(120*self.RESOLUTION_SCALE):,:])

    sign_and_decimal_horizontal = cv2.vconcat(sign_and_decimal_section)
    numbers_horizontal = cv2.vconcat(numbers_section)

    return student_id_section, exam_code_section, part1_section, part1_section_horizontal, part2_section, part2_section_horizontal,part3_section, sign_and_decimal_horizontal, numbers_horizontal

  def scan_student_id(self,student_id_section):
    #Scan for Student ID
    thresh = self.preprocess_img_threshold(student_id_section)
    filteredContours = self.find_contours(thresh)
    sorted_ctrs = self.sort_contours(filteredContours,8)
    bubbled_indices = self.get_bubbled_indicies(sorted_ctrs,thresh)

    student_id = [None]*8
    for i in (bubbled_indices):
      number = i//8
      position = i%8
      student_id[position] = number
    str_id = ''
    for i in student_id:
      str_id += str(i)
    return str_id
  def scan_exam_code(self,exam_code_section, EXAM_CODE_NUMBER=4):
    thresh = self.preprocess_img(exam_code_section)
    filteredContours = self.find_contours(thresh)
    sorted_ctrs = self.sort_contours(filteredContours,EXAM_CODE_NUMBER)
    bubbled_indices = self.get_bubbled_indicies(sorted_ctrs,thresh)
    #now handle bubbled_indices, which varies depend on which section it is
    exam_code = [None]*EXAM_CODE_NUMBER
    for i in (bubbled_indices):
      number = i//EXAM_CODE_NUMBER
      position = i%EXAM_CODE_NUMBER
      exam_code[position] = number
    str_exam_code = ''
    for i in exam_code:
      str_exam_code += str(i)
    return str_exam_code
  def scan_part1(self,part1_section_horizontal):
    thresh = self.preprocess_img(part1_section_horizontal)
    filteredContours = self.find_contours(thresh)
    sorted_ctrs = self.sort_contours(filteredContours,4)  #there are 4 answers each row
    bubbled_indices = self.get_bubbled_indicies(sorted_ctrs,thresh)
    bubbled_indices = [x + 1 for x in bubbled_indices]
    answers = {i + 1: [] for i in range(40)}
    for i in (bubbled_indices):
      answer = i%4 if i%4!=0 else 4
      alphabet_answer = chr(64+answer)
      index = i//4 if i%4==0 else i//4+1
      answers[index].append(alphabet_answer)

    part1_answers = {k: v for k, v in answers.items() if len(v) == 1}
    return part1_answers
  def scan_part2(self, part2_section_horizontal):
    thresh = self.preprocess_img_threshold(part2_section_horizontal)
    filteredContours = self.find_contours(thresh)
    sorted_ctrs = self.sort_contours(filteredContours,2)  #there are 2 answers each row
    bubbled_indices = self.get_bubbled_indicies(sorted_ctrs,thresh)
    bubbled_indices = [x + 1 for x in bubbled_indices]
    answers = {i + 1: [] for i in range(32)}
    for i in (bubbled_indices):
      answer = True if i%2!=0 else False
      index = i//2 if i%2==0 else i//2+1
      answers[index].append(answer)
    answers = {k: v for k, v in answers.items() if len(v) == 1}
    #rearrange dictionary
    part2_answers = {}
    # Group size
    group_size = 4
    # Grouping logic
    for i in range(0, len(answers), group_size):
        group_index = i // group_size + 1
        part2_answers[group_index] = {
            'A': answers.get(i + 1, []),
            'B': answers.get(i + 2, []),
            'C': answers.get(i + 3, []),
            'D': answers.get(i + 4, [])
        }
    return part2_answers
  def scan_part3(self,sign_and_decimal_horizontal, numbers_horizontal):
    part3_answers_dict = {i: [""] * 4 for i in range(1, 7)}
    part3_answers_sign_dict = {i: "" for i in range(1, 7)}
    #sign_and_decimal_horizontal
    thresh = self.preprocess_img(sign_and_decimal_horizontal)
    filteredContours = self.find_contours(thresh)
    sorted_ctrs = self.sort_contours(filteredContours,3) 
    bubbled_indices = self.get_bubbled_indicies(sorted_ctrs,thresh)
    for i in (bubbled_indices):
      index = i//3 if i%3==0 else i//3+1
      if i%3==0:
        part3_answers_sign_dict[index+1] = '-'
      else:
        part3_answers_dict[index][i%3] = '.'
    #numbers_horizontal
    thresh = self.preprocess_img(numbers_horizontal)
    filteredContours = self.find_contours(thresh)
    sorted_ctrs = self.sort_contours(filteredContours,4)  #there are 4 answers each row
    bubbled_indices = self.get_bubbled_indicies(sorted_ctrs,thresh)


    for i in (bubbled_indices):
      index = i//40
      #215//40 = 5, 215-5*40
      index_in_question = i-index*40
      number = index_in_question//4
      position = i%4
      part3_answers_dict[index+1][position] = str(number)

    part3_answers = {i: "" for i in range(1, 7)}
    for i in range(1,7):
      answer = part3_answers_sign_dict[i]+"".join(part3_answers_dict[i])
      part3_answers[i] = float(answer) 
    return part3_answers
  def save_scan_result(self,image, output_path, student_id_section, exam_code_section, part1_section, part2_section, part3_section):
    # Create a copy of the original image to draw on
    combined_image_with_sections = image.copy()

    # Ensure the demo images have the same number of channels as the original image if necessary
    combined_image_with_sections = cv2.cvtColor(combined_image_with_sections, cv2.COLOR_GRAY2BGR)


    # Get the original coordinates for each section
    student_id_coords = (int(350*self.RESOLUTION_SCALE), int(1052*self.RESOLUTION_SCALE), int(1634*self.RESOLUTION_SCALE), int(2043*self.RESOLUTION_SCALE))
    exam_code_coords = (int(354*self.RESOLUTION_SCALE), int(1056*self.RESOLUTION_SCALE), int(2093*self.RESOLUTION_SCALE), int(2310*self.RESOLUTION_SCALE))
    part1_coords = (int(1244*self.RESOLUTION_SCALE), int(1887*self.RESOLUTION_SCALE), int(248*self.RESOLUTION_SCALE), int(2254*self.RESOLUTION_SCALE))
    part2_coords = (int(2074*self.RESOLUTION_SCALE), int(2319*self.RESOLUTION_SCALE), int(248*self.RESOLUTION_SCALE), int(2232*self.RESOLUTION_SCALE))
    part3_coords = (int(2530*self.RESOLUTION_SCALE), int(3231*self.RESOLUTION_SCALE), int(254*self.RESOLUTION_SCALE), int(2238*self.RESOLUTION_SCALE))

    # Re-detect bubbled indices and draw red circles on demo images
    # Student ID section
    thresh_student_id = self.preprocess_img_threshold(student_id_section)
    filteredContours_student_id = self.find_contours(thresh_student_id)
    sorted_ctrs_student_id = self.sort_contours(filteredContours_student_id,8)
    bubbled_indices_student_id = self.get_bubbled_indicies(sorted_ctrs_student_id,thresh_student_id)
    student_id_demo = cv2.cvtColor(student_id_section.copy(), cv2.COLOR_GRAY2BGR)
    cv2.drawContours(student_id_demo, filteredContours_student_id, -1, (0, 255, 0), 2) # Draw green contours
    for i in bubbled_indices_student_id:
      cv2.drawContours(student_id_demo, [sorted_ctrs_student_id[i]], -1, (0, 0, 255), -1) # Draw red circles

    # Exam Code section
    thresh_exam_code = self.preprocess_img(exam_code_section)
    filteredContours_exam_code = self.find_contours(thresh_exam_code)
    sorted_ctrs_exam_code = self.sort_contours(filteredContours_exam_code,self.EXAM_CODE_NUMBER)
    bubbled_indices_exam_code = self.get_bubbled_indicies(sorted_ctrs_exam_code,thresh_exam_code)
    exam_code_demo = cv2.cvtColor(exam_code_section.copy(), cv2.COLOR_GRAY2BGR)
    cv2.drawContours(exam_code_demo, filteredContours_exam_code, -1, (0, 255, 0), 2) # Draw green contours
    for i in bubbled_indices_exam_code:
      cv2.drawContours(exam_code_demo, [sorted_ctrs_exam_code[i]], -1, (0, 0, 255), -1) # Draw red circles

    # Part 1 section
    thresh = self.preprocess_img(part1_section)
    filteredContours = self.find_contours(thresh)
    sorted_ctrs = self.sort_contours(filteredContours,16)
    bubbled_indices = self.get_bubbled_indicies(sorted_ctrs,thresh)
    part1_demo = cv2.cvtColor(part1_section.copy(), cv2.COLOR_GRAY2BGR)
    cv2.drawContours(part1_demo, filteredContours, -1, (0, 255, 0), 2) # Draw green contours
    for i in bubbled_indices:
      cv2.drawContours(part1_demo, [sorted_ctrs[i]], -1, (0, 0, 255), -1) # Draw red circles

      # Part 2 section
    thresh = self.preprocess_img_threshold(part2_section)
    filteredContours = self.find_contours(thresh)
    sorted_ctrs = self.sort_contours(filteredContours,16)
    bubbled_indices = self.get_bubbled_indicies(sorted_ctrs,thresh)
    part2_demo = cv2.cvtColor(part2_section.copy(), cv2.COLOR_GRAY2BGR)
    cv2.drawContours(part2_demo, filteredContours, -1, (0, 255, 0), 2) # Draw green contours
    for i in bubbled_indices:
      cv2.drawContours(part2_demo, [sorted_ctrs[i]], -1, (0, 0, 255), -1) # Draw red circles

      # Part 3 section
    thresh = self.preprocess_img(part3_section)
    filteredContours = self.find_contours(thresh)
    sorted_ctrs = self.sort_contours(filteredContours,16)
    bubbled_indices = self.get_bubbled_indicies(sorted_ctrs,thresh)
    part3_demo = cv2.cvtColor(part3_section.copy(), cv2.COLOR_GRAY2BGR)
    cv2.drawContours(part3_demo, filteredContours, -1, (0, 255, 0), 2) # Draw green contours
    for i in bubbled_indices:
      cv2.drawContours(part3_demo, [sorted_ctrs[i]], -1, (0, 0, 255), -1) # Draw red circles


    # Overlay the demo images onto the original image at their respective positions
    # Ensure the demo images have the same dimensions as the original sections before overlaying
    combined_image_with_sections[student_id_coords[0]:student_id_coords[1], student_id_coords[2]:student_id_coords[3]] = cv2.resize(student_id_demo, (student_id_coords[3]-student_id_coords[2], student_id_coords[1]-student_id_coords[0]))
    combined_image_with_sections[exam_code_coords[0]:exam_code_coords[1], exam_code_coords[2]:exam_code_coords[3]] = cv2.resize(exam_code_demo, (exam_code_coords[3]-exam_code_coords[2], exam_code_coords[1]-exam_code_coords[0]))
    combined_image_with_sections[part1_coords[0]:part1_coords[1], part1_coords[2]:part1_coords[3]] = cv2.resize(part1_demo, (part1_coords[3]-part1_coords[2], part1_coords[1]-part1_coords[0]))
    combined_image_with_sections[part2_coords[0]:part2_coords[1], part2_coords[2]:part2_coords[3]] = cv2.resize(part2_demo, (part2_coords[3]-part2_coords[2], part2_coords[1]-part2_coords[0]))
    combined_image_with_sections[part3_coords[0]:part3_coords[1], part3_coords[2]:part3_coords[3]] = cv2.resize(part3_demo, (part3_coords[3]-part3_coords[2], part3_coords[1]-part3_coords[0]))

    
    cv2.waitKey(0)
    # Display the combined image
    cv2.imwrite(output_path,combined_image_with_sections)
  


