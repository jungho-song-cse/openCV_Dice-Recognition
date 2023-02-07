import cv2

# referenced lecture slides 09_Edge Detection


def hough_circles(source):
    blurred = cv2.blur(source, (3, 3))
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 20, param1=150, param2=30)   # set minDist 20

    #출력용 코드
    dst = cv2.cvtColor(source, cv2.COLOR_GRAY2BGR)
    if circles is not None:
        for i in range(circles.shape[1]):
            cx, cy, radius = circles[0][i]
            cv2.circle(dst, (round(cx), round(cy)), round(radius), (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow('dst', dst)
    cv2.waitKey()
    cv2.destroyAllWindows()

    print(len(circles[0]))


src = cv2.imread('img1_4.png', cv2.IMREAD_GRAYSCALE)

if src is None:
    print('Image load failed!')
    exit()

hough_circles(src)
