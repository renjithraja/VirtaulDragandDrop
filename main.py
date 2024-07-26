import cv2
from cvzone.HandTrackingModule import HandDetector
import cvzone
import numpy as np

cap = cv2.VideoCapture(0)  # Use 0 for default camera
cap.set(3, 1280)
cap.set(4, 720)
detector = HandDetector(detectionCon=0.8)
colorR = (255, 0, 255)

class DragRect:
    def __init__(self, posCenter, size=[200, 200]):
        self.posCenter = posCenter
        self.size = size

    def update(self, cursor):
        cx, cy = self.posCenter
        w, h = self.size

        # If the index finger tip is in the rectangle region
        if cx - w // 2 < cursor[0] < cx + w // 2 and \
           cy - h // 2 < cursor[1] < cy + h // 2:
            self.posCenter = cursor

rectList = []
for x in range(5):
    rectList.append(DragRect([x * 250 + 150, 150]))

while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture image")
        continue  # Skip the rest of the loop and try again

    img = cv2.flip(img, 1)
    hands, img = detector.findHands(img)  # returns hands list and the image with drawings

    if hands:
        hand1 = hands[0]  # Get the first hand found
        lmList = hand1["lmList"]  # List of 21 Landmark points
        if lmList:
            # Extract the x and y coordinates of the landmarks
            x1, y1 = lmList[8][:2]  # Index finger tip
            x2, y2 = lmList[12][:2]  # Middle finger tip
            l, _, _ = detector.findDistance((x1, y1), (x2, y2), img)  # Pass x, y coordinates directly
            print(l)
            if l < 30:
                cursor = lmList[8][:2]  # index finger tip landmark (x, y)
                # call the update here
                for rect in rectList:
                    rect.update(cursor)

    ## Draw transparency
    imgNew = np.zeros_like(img, np.uint8)
    for rect in rectList:
        cx, cy = rect.posCenter
        w, h = rect.size
        cv2.rectangle(imgNew, (cx - w // 2, cy - h // 2),
                      (cx + w // 2, cy + h // 2), colorR, cv2.FILLED)
        cvzone.cornerRect(imgNew, (cx - w // 2, cy - h // 2, w, h), 20, rt=0)

    out = img.copy()
    alpha = 0.5
    mask = imgNew.astype(bool)
    out[mask] = cv2.addWeighted(img, alpha, imgNew, 1 - alpha, 0)[mask]

    cv2.imshow("Image", out)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit the loop
        break

cap.release()
cv2.destroyAllWindows()
