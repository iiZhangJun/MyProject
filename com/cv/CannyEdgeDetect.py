import cv2


image = cv2.imread('../../pic/mst.png')
image_blur = cv2.GaussianBlur(image, (3, 3), 0)
image_gray = cv2.cvtColor(image_blur, cv2.COLOR_BGR2GRAY)
cv2.namedWindow('pic', cv2.WINDOW_NORMAL)
image_binary = cv2.Canny(image_blur,100,150, 200)
cv2.imshow('pic', image_binary)
cv2.waitKey(0)

image_con, contours, hierarchy = cv2.findContours(image_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
imag = cv2.drawContours(image, contours, -1, (0,255,0), 1)

cv2.imshow('pic', imag)
cv2.waitKey(0)