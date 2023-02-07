import cv2


def hough_circles(source):  # referenced lecture slides 09 87p
    blurred = cv2.blur(source, (3, 3))
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 20, param1=150, param2=50)   # accumulation array set 50

    # 출력용 코드
    dst = cv2.cvtColor(source, cv2.COLOR_GRAY2BGR)
    if circles is not None:
        for i in range(circles.shape[1]):
            cx, cy, radius = circles[0][i]
            cv2.circle(dst, (round(cx), round(cy)), round(radius), (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imshow('source', dst)
    cv2.waitKey()
    cv2.destroyAllWindows()


    return len(circles[0])


def count_circles(source):  # referenced lecture slides 12 21p
    _, source_bin = cv2.threshold(source, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cnt, labels, stats, centroids = cv2.connectedComponentsWithStats(source_bin)
    circles_count = []  # save count of circles of objects

    for i in range(1, cnt):
        (x, y, w, h, area) = stats[i]

        if area < 20:
            continue
        circles_count.append(hough_circles(source[y:y + h, x:x + w]))  # count every objects' circles

        # 출력용 코드
        pt1 = (x,y)
        pt2=(x+w,y+h)
        cv2.rectangle(source,pt1,pt2,(255,0,255))

    cv2.imshow('source', source)
    cv2.waitKey()
    cv2.destroyAllWindows()

    circles_count.sort()  # sort counted circles
    for i in circles_count:  # print circle nums of dices
        print(i)


src = cv2.imread('img2_1.png', cv2.IMREAD_GRAYSCALE)

if src is None:
    print('Image load failed!')
    exit()

count_circles(src)
