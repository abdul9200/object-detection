import cv2
thres=0.45
cap=cv2.VideoCapture(0)
cap.set(3,648)
cap.set(4,480)
cap.set(10,70)
classLabels=[]
file_name='Labels.txt'
with open(file_name,'rt') as f:
    classLabels=f.read().rstrip('\n').split('\n')
configPath='ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPAth='frozen_inference_graph.pb'
net=cv2.dnn_DetectionModel(weightsPAth,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/127.5)
net.setInputMean(127.5)
net.setInputSwapRB(True)
classIds,confs,bbox=net.detect(img,confThreshold=0.7)
print(classIds,bbox)
if len(classIds) != 0:
    for classId,confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
        cv2.rectangle(img,box,color=(0,255,0),thickness=3)
        cv2.putText(img,classLabels[classId-1],(box[0]+10,box[1]+30),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        cv2.putText(img, str(round(confidence*100,2)), (box[0] + 200, box[1] + 30), cv2.FONT_HERSHEY_COMPLEX, 1,
                        (255, 255, 0), 2)
    cv2.imshow("Output",img)
    cv2.waitKey(0)
