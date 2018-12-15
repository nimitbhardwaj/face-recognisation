import sys
from detector.Detector import Detector
from recognizer.Recognizer import Recognizer

class Main(object):
    def __init__(self):
        pass
    def main(self, val):
        if val == "det":
            Detector().start()
        elif val == "train":
            Recognizer().train()
        elif val == "recog":
            Recognizer().recognize()
    

if __name__ == '__main__':
    m = Main()
    if len(sys.argv) == 1:
        print('Enter "detector" for image detection of "recognizer" for recognization')
    elif len(sys.argv) == 2 and sys.argv[1] == 'detector':
        m.main("det")
    elif len(sys.argv) == 2 and sys.argv[1] == 'recognizer':
        m.main("recog")
    elif len(sys.argv) == 3 and sys.argv[1] == 'recognizer' and sys.argv[2] == 'train':
        m.main("train")
