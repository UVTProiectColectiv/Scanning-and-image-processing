import cv2 as cv
import numpy as np
import utlis
from PIL import Image


def run(path):
    width_img = 700
    height_img = 700
    questions = 5
    choices = 5
    ans = ["A", "D", "B", "E", "E"]
    img = cv.imread(path)
    # print(type(img))

    # PREPROCESSING
    img = cv.resize(img, (width_img, height_img))
    # print(type(img))
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
    rect_con = utlis.rect_contour(contours)
    # print(rectCon)
    biggest_contour = utlis.get_corner_points(rect_con[0])
    code_contour = utlis.get_corner_points(rect_con[2])
    # print(biggestContour)
    # print(biggestContour.shape)
    # rect_con[1] - the second largest

    if biggest_contour.size != 0 and code_contour.size != 0:
        cv.drawContours(img_biggest_contours, biggest_contour, -1, (255, 0, 0), 30)
        cv.drawContours(img_biggest_contours, code_contour, -1, (0, 0, 255), 30)

        biggest_contour = utlis.reorder(biggest_contour)

        pt_g1 = np.float32(biggest_contour)
        pt_g2 = np.float32([[0, 0], [width_img, 0], [0, height_img], [width_img, height_img]])
        matrix = cv.getPerspectiveTransform(pt_g1, pt_g2)
        img_warp_colored = cv.warpPerspective(img, matrix, (width_img, height_img))

        # code = utlis.extract_code(img)
        # print(code)

        # APPLY THRESHOLD
        img_warp_gray = cv.cvtColor(img_warp_colored, cv.COLOR_BGR2GRAY)
        img_thresh = cv.threshold(img_warp_gray, 170, 255, cv.THRESH_BINARY_INV)[1]

        boxes = utlis.split_boxes(img_thresh)
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
            ans_added = utlis.convert(col, row)
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
        # img_result = utlis.show_answers(img_result, my_index, grading, ans, questions, choices)
        # img_raw_drawing = np.zeros_like(img_warp_colored)
        # img_raw_drawing = utlis.show_answers(img_raw_drawing, my_index, grading, ans, questions, choices)

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


def main():
    path = r'img4.jpg'
    # image = utlis.read_image(1)
    run(path)


if __name__ == '__main__':
    main()
