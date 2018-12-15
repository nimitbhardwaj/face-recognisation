import cv2
import os
import numpy as np

from detector.Detector import Detector

class Recognizer(object):
    def __init__(self):
        self.det = Detector()
    
    def train(self):
        faces = []
        recs = []
        k = 0
        while k != 4:
            img = self.__getClick()
            face, rec, valid = self.det.getClassifiedImg(img, onlyFace=True)
            if valid:
                faces.append(face)
                recs.append(rec)
                k += 1
            else:
                print('invalid face')
        name = input('Enter the name of the Individual')
        self.__storeFace(faces, name)


    def recognize(self):
        faces, labels = self.__getTrainData(os.path.join('assets', 'training_data'))
        face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        face_recognizer.train(faces, np.array(labels))
        for frame in self.__getCam():
            if type(frame) == int:
                break
            face, rec, valid = self.det.getClassifiedImg(frame, onlyFace=True)
            
            if valid:
                lab = face_recognizer.predict(cv2.cvtColor(face, cv2.COLOR_BGR2GRAY))
                self.drawRect(frame, rec, self.getNameOf(lab[0]))


            cv2.imshow("Recognizer", frame)

    def getNameOf(self, id):
        name = ''
        with open(os.path.join('assets', 'training_data', 's{}'.format(id), 'name.txt'), 'r') as f:
            name = f.read()
        return name
            
    def drawRect(self, frame, rec, lable):
            cv2.rectangle(frame, (rec[0], rec[1]), (rec[0]+rec[2], rec[1]+rec[3]), (255, 0, 0), 2)
            cv2.putText(frame, lable, (rec[0], rec[1]), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)


    def __getClick(self):
        cap = cv2.VideoCapture(0)

        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            cv2.imshow("Train", frame)
            if cv2.waitKey(1) & 0xFF == ord('c'):
                cap.release()
                cv2.destroyAllWindows()
                return frame

    def __storeFace(self, faces, name):
        lab = 0
        traningDir = os.listdir(os.path.join('assets', 'training_data'))
        while True:
            if 's{}'.format(lab) in traningDir:
                lab += 1
            else:
                break
        
        traningPath = os.path.join('assets', 'training_data', 's{}'.format(lab))
        os.mkdir(traningPath)

        for i, face in enumerate(faces):
            cv2.imwrite(os.path.join(traningPath, '0{}.jpeg'.format(i)), face)
        with open(os.path.join(traningPath, 'name.txt'), 'w') as f:
            f.write(str(name))
        
    def __getTrainData(self, trainPath):
        faces = []
        labels = []
        for dir in os.listdir(trainPath):
            lab = int(dir.replace('s', ''))
            imgPath = os.path.join(trainPath, dir)
            for im_name in os.listdir(imgPath):
                img = cv2.imread(os.path.join(imgPath, im_name))
                if img is None:
                    continue
                faces.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
                labels.append(lab)
        return faces, labels

    def __getCam(self):
        cap = cv2.VideoCapture(0)

        while(True):
            # Capture frame-by-frame
            ret, frame = cap.read()

            yield frame

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()
        yield 0