import cv2
import numpy as np
import utlis


def main():
    image = utlis.read_image()
    image.show()
    img_crop = utlis.crop_image(image)
    nume_test = utlis.extract_name(img_crop)
    img_crop.show()
    # print(nume_test)

    img = image.copy()
    img = np.asarray(img)

    choices = utlis.run(img)
    # print(choices)

    answers = utlis.extract_db(nume_test)
    # print(answers)
    grade = utlis.get_grade(choices, answers)
    # print(grade)

    # utlis.insertPhoto()


if __name__ == '__main__':
    main()
