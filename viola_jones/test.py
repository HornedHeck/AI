import cv2


cls = cv2.CascadeClassifier('/home/hornedheck/PycharmProjects/AI/cascades/anime_v1.xml')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = cls.detectMultiScale(gray, 1.25, 1)

img = cv2.imread('vj_validator.jpg', cv2.IMREAD_UNCHANGED)
target_width = 1024
dim = 1024 / img.shape[1]
img = cv2.resize(img, (1024, int(img.shape[0] * dim)))
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
cv2.imshow("Validation", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
