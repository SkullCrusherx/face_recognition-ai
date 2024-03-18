import cv2
from tkinter import Tk,Button


root = Tk()
root.title("Face Recognition")
root.geometry('400x300')
root.resizable(False, False)
cap = cv2.VideoCapture(0)
face_shape = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
root.config(background='#717fad')

def openFile():
    while True:
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detection = face_shape.detectMultiScale(
            frame,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        for (x, y, w, h) in detection:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

btn1 = Button(root,text="Start",background='#cad5fa',command=openFile)
btn1.place(x=170,y=70)


cap.release()
cv2.destroyAllWindows()

root.mainloop()
