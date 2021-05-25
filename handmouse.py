import cv2
import mediapipe as mp
import numpy as np
import time,datetime
from pynput.keyboard import Key,Controller
from multiprocessing import Process, Lock

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands #hand solution


class HandSign:
    gesture = {
        0:'fist', 1:'one', 2:'two', 3:'three', 4:'four', 5:'five',
        6:'six', 7:'rock', 8:'spiderman', 9:'yeah', 10:'ok',
    }
    numbers = {
        0:'zero', 1:'one', 2:'two', 3:'three', 4:'four', 5:'five',
        6:'ready', 9:'two', 10:'three',
    }
    rps_gesture = {0:'rock', 5:'paper', 9:'scissors'}
    def __init__(self,model):
        self.model_file=model
        self.model=self.load_model(model)
    def watch(self,detection_conf=0.8,tracking_conf=0.6,wait=2):
        
        knn=self.model
        hands=self.hand_recognition(2,detection_conf,tracking_conf)
        cap = cv2.VideoCapture(0)
        window_name="Test Hand Recognition"
        time0=round(time.time())
        time1=0
        while cap.isOpened():
            ret, img = cap.read() #video read
            if not ret:
                continue #not ready yet
            # ----------------------------------------------
            #cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
            #cv2.setWindowProperty(window_name,cv2.WND_PROP_FULLSCREEN,
            #   cv2.WINDOW_FULLSCREEN)
            img=self.preprocess_img(img)
            result=hands.process(img)
            img=self.postprocess_img(img)
            #
            width=img.shape[1]
            height=img.shape[0]
            self.print_head(img,"Taekyung Kim, KWU, 2021",x=width*0.02,y=height*0.05)
            self.print_head(img,"OK2Go Presenter",x=width*0.02,y=height*0.1)
            self.print_head(img,"Show me double V to go next!",x=width*0.02,y=height*0.15)
            self.print_head(img,"Type q to exit",x=width*0.02,y=height*0.30)
            if result.multi_hand_landmarks is not None:
                key_found=0
                time1=round(time.time())
                for res in result.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)
                for res in result.multi_hand_landmarks:
                    idx=self.get_label(knn,res)
                    text=self.gesture[idx]
                    if text=='yeah':
                        key_found+=1
                if key_found==2 and (time1-time0)>=wait:
                    keyboard.press(Key.right)
                    keyboard.release(Key.right)
                    time0=round(time.time())
                    print("[",datetime.datetime.now(),"]"," Detected.")
                    key_found=0
            cv2.imshow(window_name, img)
            if cv2.waitKey(1) == ord('q'):
                cv2.destroyAllWindows() #close the image window
                cap.release()
                break    
    def hand_recognition(self,max_num_hands=1,detection_conf=0.5,tracking_conf=0.5):
        hands = mp_hands.Hands(
            max_num_hands=max_num_hands,
            min_detection_confidence=detection_conf,
            min_tracking_confidence=tracking_conf)
        return hands
    def train_model(self,data_file,save='hand_recognition_model.xml'):
        file = np.genfromtxt(data_file, delimiter=',')
        angle = file[:,:-1].astype(np.float32)
        label = file[:, -1].astype(np.float32)
        knn = cv2.ml.KNearest_create()
        knn.train(angle, cv2.ml.ROW_SAMPLE, label)
        knn.save(save)
    def load_model(self,file_name):
        return cv2.ml.KNearest_load(file_name)
    def print_head(self,img,text,x=50,y=50):
        cv2.putText(img, 
                    text, org=(int(x),int(y)),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                    fontScale=0.5, 
                    color=(255, 255, 255),
                    thickness=1)
    def finger_vectors(self,landmarks):
        joint=np.zeros((21,3,))
        for j, lm in enumerate(landmarks): #landmark data
            joint[j] = [lm.x, lm.y, lm.z]
        N=len(landmarks)
        v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19],:] # Parent joint
        v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],:] # Child joint
        v = v2 - v1 # [20,3] #vector (x,y,z)
        v = v / np.linalg.norm(v, axis=1)[:, np.newaxis] #normalize, 길이로 나눈다. Norm
        return v
    def cal_angles(self,finger_vecs):
        angle = np.arccos(np.einsum('ij,ij->i', #row sum
            finger_vecs[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
            finger_vecs[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]
        angle = np.degrees(angle) # Convert radian to degree
        data = np.array([angle], dtype=np.float32)
        return data

    def get_label(self,knn_model,res):
        v=self.finger_vectors(res.landmark)
        data=self.cal_angles(v)
        _, results, _, _ = knn_model.findNearest(data, 3) #res,result,neighbors,distances
        try:
            idx = int(results[0][0])
        except:
            idx=None
        return idx
    def preprocess_img(self,img):
        img = cv2.flip(img, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #색공간 변경
        return img
    def postprocess_img(self,img):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img
    
if __name__=="__main__":
    keyboard=Controller()
    hand_sign=HandSign('hand_recognition_model.xml')
    hand_sign.watch(detection_conf=0.5,tracking_conf=0.5)