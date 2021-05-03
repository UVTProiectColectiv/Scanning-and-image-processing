import cv2 as cv
import numpy as np
import utlis


def main():
    path = r'img1.jpg'
    widthImg = 700
    heightImg = 700
    questions = 5
    choices = 5
    ans = [1, 2, 0, 1, 4]
    img = cv.imread(path)

    # PREPROCESSING
    img = cv.resize(img, (widthImg, heightImg))
    imgContours = img.copy()
    imgBiggestContours = img.copy()
    imgFinal = img.copy()
    imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    imgBlur = cv.GaussianBlur(imgGray, (5, 5), 1)
    imgCanny = cv.Canny(imgBlur, 10, 50)

    # FINDING ALL CONTOURS
    contours, hierarchy = cv.findContours(imgCanny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    cv.drawContours(imgContours, contours, -1, (0, 255, 0), 10)

    # FIND RECTANGLES
    rectCon = utlis.rectContour(contours)
    biggestContour = utlis.getCornerPoints(rectCon[0])
    #print(biggestContour.shape)
    gradePoints = utlis.getCornerPoints(rectCon[1])  #the second largest

    if biggestContour.size != 0 and gradePoints.size != 0:
        cv.drawContours(imgBiggestContours, biggestContour, -1, (0, 255, 0), 20)
        cv.drawContours(imgBiggestContours, gradePoints, -1, (255, 0, 0), 20)

        biggestContour = utlis.reorder(biggestContour)
        gradePoints = utlis.reorder(gradePoints)

        ptG1 = np.float32(biggestContour)
        ptG2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
        matrix = cv.getPerspectiveTransform(ptG1, ptG2)
        imgWarpColored = cv.warpPerspective(img, matrix, (widthImg, heightImg))

        pt1 = np.float32(gradePoints)
        pt2 = np.float32([[0, 0], [325, 0], [0, 150], [325, 150]])
        matrixG = cv.getPerspectiveTransform(pt1, pt2)
        imgGradeDisplay = cv.warpPerspective(img, matrixG, (325, 150))
        #cv.imshow("Grade", imgGradeDisplay)

        # APPLY THRESHOLD
        imgWarpGray = cv.cvtColor(imgWarpColored, cv.COLOR_BGR2GRAY)
        imgThresh = cv.threshold(imgWarpGray, 170, 255, cv.THRESH_BINARY_INV)[1]

        boxes = utlis.splitBoxes(imgThresh)
        #cv.imshow("Test", boxes[2])
        #print(cv.countNonZero(boxes[2]), cv.countNonZero(boxes[1]))


        # GETTING NO ZERO PIXEL VALUES OF EACH BOX
        myPixelVal = np.zeros((questions, choices)) #5 questions, 5 multiple answers
        countC = 0 #count columns
        countR = 0 #count rows

        for image in boxes:
            totalPixels = cv.countNonZero(image)
            myPixelVal[countR][countC] = totalPixels
            countC += 1
            if countC == choices:
                countR += 1
                countC = 0
        print(myPixelVal)

        # FINDING INDEX VALUES OF THE MARKINGS
        myIndex = []
        for x in range(0, questions):
            arr = myPixelVal[x]
            #print("Array: ", arr)
            myIndexVal = np.where(arr==np.amax(arr))
            #print(myIndexVal[0])
            myIndex.append(myIndexVal[0][0])
        print(myIndex)

        # GRADING
        grading = []
        for x in range(0, questions):
            if ans[x] == myIndex[x]:
                grading.append(1)
            else:
                grading.append(0)
        #print(grading)
        score = (sum(grading)/questions)*100 # FINAL GRADE
        print(score)

        # DISPLAYING ANSWERS
        imgResult = imgWarpColored.copy()
        imgResult = utlis.showAnswers(imgResult, myIndex, grading, ans, questions, choices)
        imgRawDrawing = np.zeros_like(imgWarpColored)
        imgRawDrawing = utlis.showAnswers(imgRawDrawing, myIndex, grading, ans, questions, choices)

        invMatrix = cv.getPerspectiveTransform(pt2, pt1)  # INVERSE TRANSFORMATION MATRIX
        imgInvWarp = cv.warpPerspective(imgRawDrawing, invMatrix, (widthImg, heightImg))

        imgFinal = cv.addWeighted(imgFinal, 1, imgInvWarp, 1, 0)

    #print(biggestContour)
    #print(len(biggestContour))





    imgBlank = np.zeros_like(img)
    imageArray = ([img, imgGray, imgBlur, imgCanny],
                  [imgContours, imgBiggestContours, imgWarpColored, imgThresh],
                  [imgResult, imgRawDrawing, imgInvWarp, imgFinal])
    imgStack = utlis.stackImages(imageArray, 0.3)



    cv.imshow("Stacked Images", imgStack)
    cv.waitKey(0)


if __name__ == '__main__':
    main()