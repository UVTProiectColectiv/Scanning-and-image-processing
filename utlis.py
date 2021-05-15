import cv2
import numpy as np
from pytesseract import pytesseract


# TO STACK ALL THE IMAGES IN ONE WINDOW
def stack_images(img_array, scale, lables=[]):
    rows = len(img_array)
    cols = len(img_array[0])
    rows_available = isinstance(img_array[0], list)
    width = img_array[0][0].shape[1]
    height = img_array[0][0].shape[0]
    if rows_available:
        for x in range(0, rows):
            for y in range(0, cols):
                img_array[x][y] = cv2.resize(img_array[x][y], (0, 0), None, scale, scale)
                if len(img_array[x][y].shape) == 2:
                    img_array[x][y] = cv2.cvtColor(img_array[x][y], cv2.COLOR_GRAY2BGR)
        image_blank = np.zeros((height, width, 3), np.uint8)
        hor = [image_blank]*rows
        hor_con = [image_blank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(img_array[x])
            hor_con[x] = np.concatenate(img_array[x])
        ver = np.vstack(hor)
        # ver_con = np.concatenate(hor)
    else:
        for x in range(0, rows):
            img_array[x] = cv2.resize(img_array[x], (0, 0), None, scale, scale)
            if len(img_array[x].shape) == 2:
                img_array[x] = cv2.cvtColor(img_array[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(img_array)
        # hor_con = np.concatenate(img_array)
        ver = hor
    if len(lables) != 0:
        each_img_width = int(ver.shape[1] / cols)
        each_img_height = int(ver.shape[0] / rows)
        # print(eachImgHeight)
        for d in range(0, rows):
            for c in range(0, cols):
                cv2.rectangle(ver, (c*each_img_width, each_img_height*d),
                              (c*each_img_width+len(lables[d][c])*13+27, 30+each_img_height*d),
                              (255, 255, 255), cv2.FILLED)
                cv2.putText(ver, lables[d][c], (each_img_width*c+10, each_img_height*d+20),
                            cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 255), 2)

    return ver


def rect_contour(contours):

    rect_con = []
    for i in contours:
        area = cv2.contourArea(i)
        # print(area)
        if area > 50:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            # print(len(approx))  # CORNER POINTS
            if len(approx) == 4:
                rect_con.append(i)
    rect_con = sorted(rect_con, key=cv2.contourArea, reverse=True)
    # print(len(rectCon))

    return rect_con


def get_corner_points(cont):
    peri = cv2.arcLength(cont, True)  # LENGTH OF CONTOUR
    # print(peri)
    approx = cv2.approxPolyDP(cont, 0.02 * peri, True)  # APPROXIMATE THE POLY TO GET CORNER POINTS
    # print(approx)

    return approx


def reorder(points):
    points = points.reshape((4, 2))
    new_points = np.zeros((4, 1, 2), np.int32)
    add = points.sum(1)
    # print(points)
    # print(add)
    new_points[0] = points[np.argmin(add)]  # [0, 0]
    new_points[3] = points[np.argmax(add)]  # [width, height]

    diff = np.diff(points, axis=1)

    new_points[1] = points[np.argmin(diff)]  # [width, 0]
    new_points[2] = points[np.argmax(diff)]  # [0, height]

    # print(diff)
    return new_points


def split_boxes(img):
    rows = np.vsplit(img, 5)  # vertical split
    boxes = []

    for r in rows:
        cols = np.hsplit(r, 5)  # horizontal split
        for box in cols:
            boxes.append(box)
            # cv2.imshow("Split", box)

    return boxes


def show_answers(img, my_index, grading, ans, questions, choices):
    sec_w = int(img.shape[1]/questions)  # section width
    sec_h = int(img.shape[0]/choices)  # section height

    for x in range(0, questions):
        my_ans = my_index[x]
        c_x = (my_ans*sec_w) + sec_w//2
        c_y = (x*sec_h) + sec_h//2

        if grading[x] == 1:
            my_color = (0, 255, 0)
        else:
            my_color = (0, 0, 255)
            correct_ans = ans[x]
            cv2.circle(img, ((correct_ans*sec_w)+sec_w//2, (x*sec_h)+sec_h//2), 25, (0, 255, 0), cv2.FILLED)

        cv2.circle(img, (c_x, c_y), 50, my_color, cv2.FILLED)

    return img


def extract_code(img):
    path_to_tesseract = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    pytesseract.tesseract_cmd = path_to_tesseract
    text = pytesseract.image_to_string(img)
    text = text[:-1].strip().split(" ")
    # print(type(text[-1]))
    code = int(text[-1])
    # print(code)
    # print(type(code))
    return code
