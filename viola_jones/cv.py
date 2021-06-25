import os
import subprocess

import cv2

from dataset_utils import create_info


def wccount(filename):
    out = subprocess.Popen(['wc', '-l', filename],
                           stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT
                           ).communicate()[0]
    return int(out.partition(b' ')[0])


# '/home/hornedheck/PycharmProjects/AI/models/anime_v3.xml'
def test(path: str):
    cls = cv2.CascadeClassifier(path)

    img = cv2.imread('vivy.jpg', cv2.IMREAD_UNCHANGED)
    target_width = 720
    dim = target_width / img.shape[1]
    img = cv2.resize(img, (target_width, int(img.shape[0] * dim)))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = cls.detectMultiScale(gray, 1.2, 3, minSize=[100, 100])
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.imshow("Validation", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def train_classifier(iterations: int = 15, num_pos: int = None, num_neg: int = None):
    create_info()
    pos_len = wccount('faces.info')
    if num_pos is None:
        num_pos = pos_len - iterations - 5

    if num_neg is None:
        num_neg = wccount('empty.info')

    command = f'/opt/opencv3/bin/opencv_createsamples -info faces.info -num {pos_len} -w 24 -h 24 -vec faces.vec'
    print(command)
    os.system(command)

    command = f'/opt/opencv3/bin/opencv_traincascade' \
              f' -data model' \
              f' -vec faces.vec' \
              f' -numPos {num_pos}' \
              f' -numNeg {num_neg}' \
              f' -bg empty.info' \
              f' -w 24 -h 24' \
              f' -numStages {iterations}' \
              f' -featureType HAAR'
    print(command)
    os.system(command)


if __name__ == '__main__':
    test('/home/hornedheck/PycharmProjects/AI/models/anime_v4.xml')
