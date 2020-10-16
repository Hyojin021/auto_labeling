import numpy as np
import cv2
import os

def process(path):
    origin_im = cv2.imread(path)

    # crop_im = origin_im[730:900, :]

    draw_im = origin_im.copy()
    gray_im = cv2.cvtColor(origin_im, cv2.COLOR_BGR2GRAY)

    _, thr = cv2.threshold(gray_im, 180, 255, cv2.THRESH_BINARY_INV)

    kernel = np.ones((3, 3), np.uint8)
    thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel, iterations=3)

    contours, _ = cv2.findContours(thr, cv2.RETR_TREE, cv2.THRESH_BINARY_INV)

    draw_cnt = 0
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        perimeter = 1.0 * cv2.arcLength(cnt, True)

        if area < 700:
            continue
        cv2.drawContours(draw_im, [cnt], -1, (0, 0, 255), -1)
        draw_cnt += 1


    cv2.namedWindow('thr', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('thr', 1500, 300)
    cv2.imshow('thr', thr)

    cv2.namedWindow('draw_im', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('draw_im', 1500, 300)
    cv2.imshow('draw_im', draw_im)

    cv2.namedWindow('origin_im', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('origin_im', 1500, 300)
    cv2.imshow('origin_im', gray_im)

    cv2.waitKey(0)
    cv2.destroyAllWindows()




if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser('machine_vision')
    parser.add_argument('--input', '-i', default='/media/jsk/data/namwon/sealing/one_reclassification/TMP_OK/img')
    args = parser.parse_args()

    input_paths = [os.path.join(args.input, f.name) for f in os.scandir(args.input)]
    # random.shuffle(input_paths)
    for input_path in input_paths:
        print(input_path)
        process(input_path)
