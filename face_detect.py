import dlib 
import cv2 
import imutils 
import time


stream = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()


def draw_border(img, pt1, pt2, color, thickness, r, d):
    x1, y1 = pt1
    x2, y2 = pt2

    cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)

    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)

    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)

    cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)
    
count = 0

while True:
    if count % 3 != 0:
        (grabbed, frame) = stream.read()

        frame = imutils.resize(frame, width=1080, height=1080)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        overlay = frame.copy()
        output = frame.copy()

        alpha  = 0.5

        face_rects = detector(gray, 0)

        for i, d in enumerate(face_rects):
            x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()

            draw_border(overlay, (x1, y1), (x2, y2), (0, 0, 255), 2, 5, 30)

        cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

        cv2.imshow("Face Detection", output)
        key = cv2.waitKey(1) & 0xFF
        
    count +=1

fourcc = cv2.VideoWriter_fourcc(*'H264')
out = cv2.VideoWriter('appsrc ! queue ! videoconvert ! video/x-raw ! omxh264enc ! video/x-h264 ! h264parse ! rtph264pay ! udpsink host=localhost port=5000 sync=false',0,25.0,(640,480))
# out = cv2.VideoWriter('output.avi', fourcc, 20, (640, 480), True)


    # press q to break out of the loop
    # if key == ord("q"):
    #     break
        


cv2.destroyAllWindows()
stream.stop()