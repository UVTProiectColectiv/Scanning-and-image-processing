import numpy as np
import utlis


def main():
    # path = r'img4.jpg'
    image = utlis.read_image(1)
    # print(type(image)) <class 'PIL.JpegImagePlugin.JpegImageFile'>
    # image.show()
    image = np.asarray(image)
    # cv.imshow("Test", image)
    # print(type(image))
    utlis.run(image)


if __name__ == '__main__':
    main()
