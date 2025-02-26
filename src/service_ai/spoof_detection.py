import sys
from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
sys.path.append(str(FILE.parents[1]))
if str(ROOT) not in sys.path:
	sys.path.append(str(ROOT))

from libs.base_libs import *
from torchvision import transforms
from linformer import Linformer
from vit_pytorch.efficient import ViT

class Core: 
	def __init__(self,home,conf) -> None:
		efficient_transformer = Linformer(
			dim=128,
			seq_len=49+1,  # 7x7 patches + 1 cls-token
			depth=12,
			heads=8,
			k=64
		)

		self._model = ViT(
			dim=128,
			image_size=conf['IM_SIZE'],
			patch_size=64,
			num_classes=2,
			transformer=efficient_transformer,
			channels=3,
		).to(conf['DEVICE'])

		self._model.load_state_dict(torch.load(os.path.join(home,conf['WEIGHT_PATH']), map_location=torch.device(conf['DEVICE'])))
		print('Initialized model')

	def softmax(self, x):
		return np.exp(x)/np.sum(np.exp(x),axis=1, keepdims=True)

	def predict(self,im):
		ops = self._model(im)
		ops = self.softmax(ops.cpu().detach().numpy())
		# print(ops)
		return ops[0]
		# predicted_cls_id=ops.argmax(dim=1)
		# return predicted_cls_id[0].item()

class SpoofDetectionRunnable:
	def __init__(self, model_path='./weights/spoofing.pt', imgsz=448, device="cuda", cls_names=['authentic', 'fake']):
		self.conf = {'DEVICE': device, 'IM_SIZE': imgsz, 'WEIGHT_PATH': model_path, 'CLS_NAMES': cls_names}

		self.core = Core("", self.conf)  # Initialize model (Only do this ONCE!!!)
		# self.cls_id_intrpr = ClassIDInterpreter(self.conf)
		print(' * Loading idcard cropper with out 4 corners model {}... Done!'.format(self.conf['WEIGHT_PATH']))

	def preProcess(self, image):
		compose=transforms.Compose(
			[
				transforms.Resize((self.conf['IM_SIZE'], self.conf['IM_SIZE'])),
				transforms.ToTensor(),
			]
		)
		device=self.conf['DEVICE']
		return torch.stack([compose(image)]).to(device)

	def inference(self, images):
		confs = []
		for image in images:
			image  = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
			p_image = Im.fromarray(image)
			preprocessed_im = self.preProcess(p_image)
			conf = self.core.predict(preprocessed_im)
			confs.append(conf)
		return confs


if __name__ == '__main__':
	test_image_path = "./spoofing_image"
	test_images = os.listdir(test_image_path)
	detector= SpoofDetectionRunnable()
	for path in test_images:
	  image = cv2.imread(os.path.join(test_image_path, path))
	  #print(image.shape)
	  #exit()
	  print(f"{path} is predicted to be {detector.inference(image)}")