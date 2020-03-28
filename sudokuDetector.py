import sys;
import numpy as np;
import cv2
from solver import solve;
from base64 import b64encode


debugNumbers = False
record = False

# Required genTestData1 to be run before, to generate Training Data

# train model
numbers = np.loadtxt('numbers.data', np.float32)
labels = np.loadtxt('labels.data', np.float32)
labels = labels.reshape((labels.size, 1))

model = cv2.ml.KNearest_create()
model.train(numbers,cv2.ml.ROW_SAMPLE, labels)

# capture
if record == True:
    cam = cv2.VideoCapture(0)

    # cam.set(cv2.CV_CAP_PROP_FRAME_WIDTH,1000);
    # cam.set(cv2.CV_CAP_PROP_FRAME_HEIGHT,100);

    cv2.namedWindow("test")

    img_counter = 0

    while True:
        ret, frame = cam.read()
        # cv2.imshow("test", frame)
        if not ret:
            break
        k = cv2.waitKey(1)

        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k%256 == 32:
            # SPACE pressed
            img_name = "opencv_frame_{}.png".format(img_counter)
            cv2.imwrite(img_name, frame)
            print("{} written!".format(img_name))
            img_counter += 1
            break

    cam.release()

    cv2.destroyAllWindows()

def solveSud(challenge):
    if record == True:
        challenge = frame
    img_grey = cv2.cvtColor(challenge, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(img_grey, (11, 11), 0)
    # cv2.imshow("blur", blur)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 2)
    # cv2.imshow("thresh", thresh)
    inv = cv2.bitwise_not(thresh)
    # cv2.imshow("inv", inv)

    draw = challenge.copy();

    contours, hierarchy = cv2.findContours(inv, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(draw, contours, -1, color=(255, 0, 0))

    # Find biggest cont
    sudCnt = None;
    for cnt in contours:
        circumference = cv2.arcLength(contours[0], True)
        if (cv2.contourArea(cnt) > 1000):

            # if len(cv2.approxPolyDP(cnt, 0.02 * circumference, True)) == 4:   # TODO:
            if (sudCnt is not None):
                if cv2.contourArea(cnt) > cv2.contourArea(sudCnt):
                    sudCnt = cnt
            else:
                sudCnt = cnt


    [x,y,w,h] = cv2.boundingRect(sudCnt)
    cv2.rectangle(draw, (x,y), (x+w, y+h), (0, 0, 255))

    if (sudCnt is None):
        print("NOPE")
        sys.exit();

    area = cv2.contourArea(sudCnt);
    circumference = cv2.arcLength(sudCnt, True);
    approx = cv2.approxPolyDP(sudCnt, 0.02 * circumference, True); # geschätztes polygon, 2. paramter gibt genauigkeit an

    # cv2.line(draw, (approx[0][0][0], approx[0][0][1]), (approx[1][0][0], approx[1][0][1]), (0, 255, 0), 1);
    # cv2.line(draw, (approx[1][0][0], approx[1][0][1]), (approx[2][0][0], approx[2][0][1]), (0, 255, 0), 1);
    # cv2.line(draw, (approx[2][0][0], approx[2][0][1]), (approx[3][0][0], approx[3][0][1]), (0, 255, 0), 1);
    # cv2.line(draw, (approx[3][0][0], approx[3][0][1]), (approx[0][0][0], approx[0][0][1]), (0, 255, 0), 1);

    # approx oder is not always the same
    # order approx to TopLeft, TopRight, BottomRight, BottomLeft
    def orderEdges(approx):
        approx = approx.reshape((4,2)) # make og array (4, 1, 2) to new array (4, 2)
        order = np.zeros((4,2),dtype = np.float32)

        add = approx.sum(1)
        order[0] = approx[np.argmin(add)]
        order[2] = approx[np.argmax(add)]
        
        diff = np.diff(approx,axis = 1)
        order[1] = approx[np.argmin(diff)]
        order[3] = approx[np.argmax(diff)]

        return order;

    approx = orderEdges(approx)
    h = np.array([ [0,0],[459,0],[459,459],[0,459] ], np.float32) # creates array
    retval = cv2.getPerspectiveTransform(approx, h)
    wrap = cv2.warpPerspective(challenge, retval, (460, 460));
    out = wrap
    wrap = wrap[5:455, 5:455]

    deltaY = wrap.shape[0] // 9
    deltaX = wrap.shape[1] // 9

    def sliceGrid():
        grid = [];
        for i in range(9):
            for j in range(9):
                grid.append(wrap[i*deltaX:(i+1)*deltaX, j*deltaY:(j+1)*deltaY])

        return grid;

    # for i in range(1, 9):
    #     cv2.line(wrap, (0, deltaY * i), (449, deltaY * i), (0, 0, 255), 2)
    #     cv2.line(wrap, (deltaX * i, 0), (deltaX * i, 449), (0, 0, 255), 2)

    grid = sliceGrid();
    sudoku = np.zeros(81) # data_struc: 1. Spalte 0, 1, 2, ... / 2. Spalte 9, 10, ...

    # cv2.imshow("sudoku", wrap)
    # cv2.waitKey(0)

    # Recognize Digits
    for i in range(len(grid)):
        num = grid[i]

        # cv2.imshow("org", num)

        num = num[10:num.shape[0] - 10, 10:num.shape[1] - 10, ]
        
        num1 = num.copy()

        testGray = cv2.cvtColor(num, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(testGray, 255, 1, 1, 15, 2)

        contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(num1, contours, -1, (0, 0, 255))

        foundNumber = False
        for cnt in contours:
            if cv2.contourArea(cnt) > 50:
                [x, y, w, h] = cv2.boundingRect(cnt)
                cv2.rectangle(num1, (x,y), (x+w,y+h), (255, 0, 0))

                if h > 15:
                    imNum = thresh[y:y+h, x:x+w]
                    imNumSm = cv2.resize(imNum, (20, 20))
                    numRes = imNumSm.reshape((1, 400))
                    numRes = np.float32(numRes)

                    retval, results, neigh_resp, dists = model.findNearest(numRes, k = 1)
                    res = int((results[0][0]))
                    
                    sudoku[i] = res
                    foundNumber = True;

                    if debugNumbers == True:
                        key = cv2.waitKey(0)
        
        if foundNumber is False:
            sudoku[i] = 0


    sudokuStr = ''
    sudoku = sudoku.astype(np.int64)
    print(sudoku)
    for i in sudoku:
        sudokuStr += str(int(i))
    result = solve(sudokuStr)

    imgenc = cv2.imencode(".jpg", out)[1]
    img_as_b64 = b64encode(imgenc)

    return (result, img_as_b64, sudoku)

