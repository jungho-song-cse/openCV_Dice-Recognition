import cv2


def hough_circles(source):  # referenced lecture slides 09 87p
    source = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(source, cv2.HOUGH_GRADIENT, 1, 5, param1=160, param2=33, maxRadius=20)

    # 출력용 코드
    dst = cv2.cvtColor(source, cv2.COLOR_GRAY2BGR)
    if circles is not None:
       for i in range(circles.shape[1]):
            cx, cy, radius = circles[0][i]
            cv2.circle(dst, (round(cx), round(cy)), round(radius), (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imshow('source', dst)
    cv2.waitKey()
    cv2.destroyAllWindows()

    if circles is None:
        return 0
    else:
        return len(circles[0])


# count_circles는 이진영상 binary_source와 원본영상 src를 받는다
def count_circles(binary_source, src):  # referenced lecture slides 12 21p
    cnt, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_source)
    circles_count = []  # save count of circles of objects

    #출력용 코드
    binary_source = cv2.cvtColor(binary_source, cv2.COLOR_GRAY2BGR)

    src = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)

    for i in range(1, cnt):
        (x, y, w, h, area) = stats[i]

        if area < 2000 or area > 50000:   # 노이즈 무시하기 위해 사용
            continue

        circles_count.append(hough_circles(src[y:y + h, x:x + w]))  # src의 객체가 있는 위치에서 원의 개수를 세어 circles_count 배열에 저장한다.

    #출력용 코드
        pt1 = (x,y)
        pt2=(x+w,y+h)
        cv2.rectangle(binary_source,pt1,pt2,(255,0,255))
    cv2.imshow('binary_source', binary_source)
    cv2.waitKey()
    cv2.destroyAllWindows()

    circles_count.sort()  # sort counted circles
    for i in circles_count:  # print circle nums of dices
        print(i)


src = cv2.imread('img5_7.png', cv2.IMREAD_GRAYSCALE)

if src is None:
    print('Image load failed!')
    exit()

src = cv2.blur(src, (5, 5))

dst = cv2.adaptiveThreshold(src, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 41, 1)

count_circles(dst, src)
