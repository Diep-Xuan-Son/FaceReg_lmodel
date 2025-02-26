import threading
import onnxruntime
import cv2
import numpy as np

class BaseOnnx(threading.Thread):
    def __init__(self, path):
        threading.Thread.__init__(self)
        devices = onnxruntime.get_available_providers()
        if 'CUDAExecutionProvider' in devices:
            devices = ['CPUExecutionProvider', 'CUDAExecutionProvider']
        else:
            devices = ['CPUExecutionProvider']
        self.session = onnxruntime.InferenceSession(path, providers=devices)

    def pre(self, input_feed):
        for key in input_feed:
            input_feed[key] = input_feed[key].astype(np.float32)
        return input_feed

    def pos(self, out):
        return out[0][0]
    
    def infer(self, input_feed):
        out = self.session.run(None, input_feed=input_feed)
        return out
    
    def __call__(self, input_feed):
        input_feed = self.pre(input_feed)
        out = self.infer(input_feed)
        out = self.pos(out)
        return out

class FakeFace:
    def __init__(self, p_onnx):
        self.model = BaseOnnx(p_onnx)
    
    def pre(self, img0):
        img0 = cv2.resize(img0, (80, 80))
        img0 = img0.transpose(2, 0, 1)
        return img0[None, ...]

    def softmax_stable(self, x):
        return(np.exp(x - np.max(x)) / np.exp(x - np.max(x)).sum())

    def post(self, o):
        r = self.softmax_stable(o)
        o = np.array([(r[0]+r[2])*0.5, r[1]*0.5])
        o = o/o.sum()
        return o

    def inference(self, imgs):
        outs = []
        for img in imgs:
            img = self.pre(img)
            o = self.model({"input.1":img})
            o = self.post(o)
            o = list(reversed(o))
            outs.append(o)
        return outs
