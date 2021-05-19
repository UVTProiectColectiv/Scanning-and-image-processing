import numpy as np
import utlis
from PIL import Image


def main():
    image = utlis.read_image(1)
    img_crop = utlis.crop_image(image)
    code = utlis.extract_code(img_crop)
    # print(code)

    img = image.copy()
    img = np.asarray(img)

    #code = utlis.extract_code(image)
    #print(code)
    choices = utlis.run(img)
    print(choices)

    grade = utlis.get_grade(choices, ['A', 'C', 'C', 'E', 'E'])
    print(grade)

    # utlis.insertPhoto()



if __name__ == '__main__':
    main()
