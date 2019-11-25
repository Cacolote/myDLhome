import cv2
import numpy as np


def detect(image):
    # 灰度化
    I2 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(I2, (3, 3), 0)

    a = 2
    O = blurred * float(a)
    O[O > 255] = 255
    O = np.round(O)
    O = O.astype(np.uint8)

    kernel = np.ones((4, 4), np.uint8)
    kernel2 = np.ones((2, 1), np.uint8)
    kernel3 = cv2.getStructuringElement(cv2.MORPH_CROSS, (2, 2))
    kernel4 = np.ones((4, 1), np.uint8)
    kernel5 = np.ones((23, 23), np.uint8)
    kernel6 = np.ones((18, 18), np.uint8)

    # 边缘提取
    opening1 = cv2.morphologyEx(O, cv2.MORPH_OPEN, kernel)
    dilation1 = cv2.dilate(opening1, kernel, iterations=1)
    closing1 = cv2.morphologyEx(opening1, cv2.MORPH_CLOSE, kernel)
    edge = dilation1 - closing1

    # otsu阈值二值化
    ret, th = cv2.threshold(edge, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 滤波
    closing2 = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel2)
    opening2 = cv2.morphologyEx(closing2, cv2.MORPH_OPEN, kernel2)

    # 提取连通区域
    erosion1 = cv2.erode(opening2, kernel4, iterations=1)
    closing3 = cv2.morphologyEx(erosion1, cv2.MORPH_CLOSE, kernel5)

    # 去除小目标
    erosion2 = cv2.erode(closing3, kernel6, iterations=1)
    dilation2 = cv2.dilate(erosion2, kernel6, iterations=1)

    # 查找图像边缘整体形成的矩形区域，可能有很多，车牌就在其中一个矩形区域中
    _, contours, hierarchy = cv2.findContours(dilation2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    temp_contours = []
    for contour in contours:
        if 15000 > cv2.contourArea(contour) > 2000:
            temp_contours.append(contour)
    car_plates = []
    for temp_contour in temp_contours:
        rect_tupple = cv2.minAreaRect(temp_contour)
        rect_width, rect_height = rect_tupple[1]
        if rect_width < rect_height:
            rect_width, rect_height = rect_height, rect_width
        aspect_ratio = rect_width / rect_height
        # 车牌正常情况下宽高比在2 - 5.5之间
        if aspect_ratio > 2 and aspect_ratio < 5.5:
            car_plates.append(temp_contour)
            rect_vertices = cv2.boxPoints(rect_tupple)
            rect_vertices = np.int0(rect_vertices)
    if len(car_plates) == 1:
        for car_plate in car_plates:
            row_min, col_min = np.min(car_plate[:, 0, :], axis=0)
            row_max, col_max = np.max(car_plate[:, 0, :], axis=0)
            cv2.rectangle(image, (row_min, col_min), (row_max, col_max), (0, 255, 0), 2)
            card = image[col_min:col_max, row_min:row_max, :]
            cv2.imshow("img", image)
        cv2.imshow("card_img.jpg", card)


if __name__ == '__main__':
    image = cv2.imread('lu.jpg')
    detect(image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
