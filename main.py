# import cv2
# import torch
# import numpy as np

# path = 'C:/Documents/PKL/yolov5/best.pt'  # Update the correct path to your model

# model = torch.hub.load('ultralytics/yolov5', 'custom',path, force_reload=True)

# #model = torch.load(path, map_location=torch.device('cpu')) 


# cap=cv2.VideoCapture('vidcoba1.mp4')
# count=0
# while True:
#     ret,frame=cap.read()
#     if not ret:
#         break
#     count +=1
#     if count % 3!=0:
#         continue
#     frame=cv2.resize(frame,(1020,600))
#     results=model(frame)
#     frame=np.squeeze(results.render())
    
#     results=model(frame)
#     cv2.imshow("FRAME",frame)
#     if cv2.waitKey(1)&0xFF==27:
#         break
# cap.release()
# cv2.destroyAllWindows()


#REALTIME DETECTION

import cv2
import torch
import numpy as np

path = 'C:/Documents/PKL/yolov5/runs/train/exp2/weights/best.pt'  # Update the correct path to your model

model = torch.hub.load('ultralytics/yolov5', 'custom', path, force_reload=True)

cap = cv2.VideoCapture(0)  # Menggunakan indeks 0 untuk webcam default

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (1020, 600))
    
    results = model(frame)
    frame = np.squeeze(results.render())
    
    cv2.imshow("Real-time Object Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Tekan tombol 'Esc' untuk keluar
        break

cap.release()
cv2.destroyAllWindows()
