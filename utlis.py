import cv2 as cv
import numpy as np
from pytesseract import pytesseract
import mysql.connector
import base64
from PIL import Image
import io


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
                img_array[x][y] = cv.resize(img_array[x][y], (0, 0), None, scale, scale)
                if len(img_array[x][y].shape) == 2:
                    img_array[x][y] = cv.cvtColor(img_array[x][y], cv.COLOR_GRAY2BGR)
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
            img_array[x] = cv.resize(img_array[x], (0, 0), None, scale, scale)
            if len(img_array[x].shape) == 2:
                img_array[x] = cv.cvtColor(img_array[x], cv.COLOR_GRAY2BGR)
        hor = np.hstack(img_array)
        # hor_con = np.concatenate(img_array)
        ver = hor
    if len(lables) != 0:
        each_img_width = int(ver.shape[1] / cols)
        each_img_height = int(ver.shape[0] / rows)
        # print(eachImgHeight)
        for d in range(0, rows):
            for c in range(0, cols):
                cv.rectangle(ver, (c*each_img_width, each_img_height*d),
                             (c*each_img_width+len(lables[d][c])*13+27, 30+each_img_height*d),
                             (255, 255, 255), cv.FILLED)
                cv.putText(ver, lables[d][c], (each_img_width*c+10, each_img_height*d+20),
                           cv.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 255), 2)

    return ver


def rect_contour(contours):

    rect_con = []
    for i in contours:
        area = cv.contourArea(i)
        # print(area)
        if area > 50:
            peri = cv.arcLength(i, True)
            approx = cv.approxPolyDP(i, 0.02 * peri, True)
            # print(len(approx))  # CORNER POINTS
            if len(approx) == 4:
                rect_con.append(i)
    rect_con = sorted(rect_con, key=cv.contourArea, reverse=True)
    # print(len(rectCon))

    return rect_con


def get_corner_points(cont):
    peri = cv.arcLength(cont, True)  # LENGTH OF CONTOUR
    # print(peri)
    approx = cv.approxPolyDP(cont, 0.02 * peri, True)  # APPROXIMATE THE POLY TO GET CORNER POINTS
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
            # cv.imshow("Split", box)

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
            cv.circle(img, ((correct_ans*sec_w)+sec_w//2, (x*sec_h)+sec_h//2), 25, (0, 255, 0), cv.FILLED)

        cv.circle(img, (c_x, c_y), 50, my_color, cv.FILLED)

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


def read_image(cod_user):

    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="112322123",
        database="proiectcolectiv"  # Name of the database
    )

    cursor = conn.cursor()
    query = 'SELECT POZA_TEST FROM tests WHERE COD_USER=%s'
    cursor.execute(query, (cod_user,))
    data = cursor.fetchall()

    image = data[0][0]
    binary_data = base64.b64decode(image)
    image = Image.open(io.BytesIO(binary_data))
    # print(type(image))
    # image.show()
    return image


def convert(x, y):
    if x == 0 and y in range(0, 5):
        return "A"
    elif x == 1 and y in range(0, 5):
        return "B"
    elif x == 2 and y in range(0, 5):
        return "C"
    elif x == 3 and y in range(0, 5):
        return "D"
    elif x == 4 and y in range(0, 5):
        return "E"


def run(img):
    questions = 5
    choices = 5
    ans = ["A", "D", "B", "E", "E"]
    # img = cv.imread(path)
    # print(type(img))

    # PREPROCESSING
    width_img = 700
    height_img = 700
    img_contours = img.copy()
    img_biggest_contours = img.copy()
    # analysing grayscale image is about 3 times faster
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Gaussian Blur filter before edge detection aims to reduce the level of noise in the image
    # which improves the result of the following edge-detection algorithm.
    img_blur = cv.GaussianBlur(img_gray, (5, 5), 1)
    img_canny = cv.Canny(img_blur, 10, 50)

    # FINDING ALL CONTOURS
    contours, hierarchy = cv.findContours(img_canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    # print(contours)
    # print(hierarchy)
    cv.drawContours(img_contours, contours, -1, (0, 255, 0), 10)

    # FIND RECTANGLES
    rect_con = rect_contour(contours)
    # print(rectCon)
    biggest_contour = get_corner_points(rect_con[0])
    code_contour = get_corner_points(rect_con[2])
    # print(biggestContour)
    # print(biggestContour.shape)
    # rect_con[1] - the second largest

    if biggest_contour.size != 0 and code_contour.size != 0:
        cv.drawContours(img_biggest_contours, biggest_contour, -1, (255, 0, 0), 30)
        cv.drawContours(img_biggest_contours, code_contour, -1, (0, 0, 255), 30)

        biggest_contour = reorder(biggest_contour)

        pt_g1 = np.float32(biggest_contour)
        pt_g2 = np.float32([[0, 0], [width_img, 0], [0, height_img], [width_img, height_img]])
        matrix = cv.getPerspectiveTransform(pt_g1, pt_g2)
        img_warp_colored = cv.warpPerspective(img, matrix, (width_img, height_img))

        # code = extract_code(img)
        # print(code)

        # APPLY THRESHOLD
        img_warp_gray = cv.cvtColor(img_warp_colored, cv.COLOR_BGR2GRAY)
        img_thresh = cv.threshold(img_warp_gray, 170, 255, cv.THRESH_BINARY_INV)[1]

        boxes = split_boxes(img_thresh)
        # cv.imshow("Test", boxes[2])
        # print(cv.countNonZero(boxes[2]), cv.countNonZero(boxes[1]))

        # GETTING NO ZERO PIXEL VALUES OF EACH BOX
        my_pixel_val = np.zeros((questions, choices))  # 5 questions, 5 multiple answers
        count_c = 0  # count columns
        count_r = 0  # count rows
        # print(my_pixel_val)

        for image in boxes:
            total_pixels = cv.countNonZero(image)
            my_pixel_val[count_r][count_c] = total_pixels
            count_c += 1
            if count_c == choices:
                count_r += 1
                count_c = 0
        # print(my_pixel_val)

        # FINDING ANSWERS VALUES OF THE MARKINGS
        my_index = []

        for x in range(0, questions):
            maxim = 0
            for y in range(0, choices):
                if my_pixel_val[x][y] > maxim:
                    maxim = my_pixel_val[x][y]
                    row = x
                    col = y
            # print(row, col)
            ans_added = convert(col, row)
            # print(ans_added)
            my_index.append(ans_added)
        print(my_index)

        # GRADING
        grading = []
        for x in range(0, questions):
            if ans[x] == my_index[x]:
                grading.append(1)
            else:
                grading.append(0)
        # print(grading)
        score = (sum(grading) / questions) * 100  # FINAL GRADE
        print(score)

        # DISPLAYING ANSWERS
        # img_result = img_warp_colored.copy()
        # img_result = show_answers(img_result, my_index, grading, ans, questions, choices)
        # img_raw_drawing = np.zeros_like(img_warp_colored)
        # img_raw_drawing = show_answers(img_raw_drawing, my_index, grading, ans, questions, choices)

    # print(biggest_contour)
    # print(len(biggest_contour))

    # img_blank = np.zeros_like(img)
    # image_array = ([img, img_gray, img_blur, img_canny], [img_contours, img_biggest_contours, img_warp_colored,
    #                                                      img_thresh],
    #               [img_result, img_raw_drawing, img_blank, img_blank])

    # img_stack = utlis.stack_images(image_array, 0.3)

    # cv.imshow("Stacked Images", img_stack)

    # DE-ALLOCATE ANY ASSOCIATED MEMORY USAGE
    if cv.waitKey(0) and 0xff == 27:
        cv.destroyAllWindows()
