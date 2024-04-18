#Data Preparation
import cv2
import numpy as np
import os
#Data
dataset_path="C:\\Users\\SHAMBHAVI MISHRA\\Desktop\\face_detection\\data\\"
facedata=[]
labels=[]
classId=0
nameMap={}
for f in os.listdir(dataset_path):
    if f.endswith(".npy"):
        nameMap[classId]=f[:-4] #get all letters other than last 4.. (.npy)
        #X-value..load face data
        dataItem=np.load(dataset_path+f)
        m=dataItem.shape[0]
        #print(dataItem.shape)
        facedata.append(dataItem)

        #Y-values..labels
        target=classId *np.ones((m,)) # 1st person getting label as 0 for i number of images in data file
        classId+=1 # from label 0 it gets incremented depending on number of people
        labels.append(target)

#list ...
"""print(facedata)
print(labels)"""
# we dont want list we want to join/stack them
XT=np.concatenate(facedata,axis=0)
yT=np.concatenate(labels,axis=0).reshape((-1,1))
print(XT.shape) #(45, 30000) ..45 rows in training data and 30000 features each
print(yT.shape) #(45, 1)...43 labels 1 single col
print(nameMap)


#given a new image we have to make predictions
#algorithm
def dist(p,q):
  return np.sqrt(np.sum((p-q)**2))
def knn(X,y,xt,k=5):
    m=X.shape[0]
    dlist=[]
    for i in range(m):
        d=dist(X[i],xt)
        dlist.append((d,y[i]))  #y[i]=0,1,2 sorting is going to be based on d
    dlist=sorted(dlist)
    darray = np.zeros((len(dlist), 2))

    for i, (d, label) in enumerate(dlist):
        darray[i, 0] = d
        darray[i, 1] = label

    # Take the first k elements
    darray = darray[:k]

    # Extract labels
    labels = darray[:, 1].astype(int)

    # Find the most common label
    unique_labels, counts = np.unique(labels, return_counts=True)
    pred = unique_labels[np.argmax(counts)]

    return pred

#predictions
# Check if the cascade file exists
cam = cv2.VideoCapture(0)
cascade_file = "haarcascade_frontalface_alt.xml"
if not os.path.exists(cascade_file):
    print("Cascade file not found!")
    exit(1)
offset = 20

model = cv2.CascadeClassifier(cascade_file)
while True:
    success, img = cam.read()
    if not success:
        print("Reading failed")
        break

    # Detect faces in the grayscale image
    faces = model.detectMultiScale(img, 1.3, 5)
    # render a box around face and predicts its name
    for f in faces:
        x, y, w, h = f
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Crop and resize the face region
        cropped_face = img[y - offset:y + h + offset, x - offset:x + w + offset]
        cropped_face = cv2.resize(cropped_face, (100, 100))
        #cv2.imshow("Image window", img)
        #predict the name usking knn
        classPredicted=knn(XT,yT,cropped_face.flatten())
        #name
        namePredicted=nameMap[classPredicted]
        #display the name box
        cv2.putText(img,namePredicted,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.imshow("Prediction window",img)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()