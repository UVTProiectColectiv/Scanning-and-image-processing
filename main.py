import numpy as np
import utlis
from PIL import Image


def main():
    image = utlis.read_image(3)
    img_crop = utlis.crop_image(image)
    code = utlis.extract_code(img_crop)
    # print(code)

    img = image.copy()
    img = np.asarray(img)

    choices = utlis.run(img)
    # print(choices)

    answers = utlis.extract_db(1005)
    print(answers)
    # grade = utlis.get_grade(choices, ['A', 'B', 'C', 'E', 'A'])
    # print(grade)

    # utlis.insertPhoto()


if __name__ == '__main__':
    main()
