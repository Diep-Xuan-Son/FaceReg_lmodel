import sys 
from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
sys.path.append(str(FILE.parents[1]))
if str(ROOT) not in sys.path:
	sys.path.append(str(ROOT))

from libs.base_libs import *
# from libs.yolo_utils import select_device
from service_ai.models_retinaface.retinaface import RetinaFace
from libs.face_preprocess import preprocess as face_preprocess

class Dets(BaseModel):
	loc: List[list] = []
	landms: List[list] = []

class PriorBox(object):
	def __init__(self, min_sizes=[[16,32],[64,128],[256,512]], steps=[8,16,32], clip=False, image_size=None, phase='train'):
		super(PriorBox, self).__init__()
		self.min_sizes = min_sizes
		self.steps = steps
		self.clip = clip
		self.image_size = image_size
		self.feature_maps = [[ceil(self.image_size[0]/step), ceil(self.image_size[1]/step)] for step in self.steps]
		self.name = "s"

	def forward(self):
		anchors = []
		for k, f in enumerate(self.feature_maps):
			min_sizes = self.min_sizes[k]
			for i, j in product(range(f[0]), range(f[1])):
				for min_size in min_sizes:
					s_kx = min_size / self.image_size[1]
					s_ky = min_size / self.image_size[0]
					dense_cx = [x * self.steps[k] / self.image_size[1] for x in [j + 0.5]]
					dense_cy = [y * self.steps[k] / self.image_size[0] for y in [i + 0.5]]
					for cy, cx in product(dense_cy, dense_cx):
						anchors += [cx, cy, s_kx, s_ky]
		# back to torch land
		# output = torch.Tensor(anchors).view(-1, 4)
		output = np.array(anchors).reshape(-1, 4)
		if self.clip:
			# output.clamp_(max=1, min=0)
			output = np.clip(output, a_max=1, a_min=0)
		return output

class RetinanetRunnable():
	def __init__(self, model_path, min_sizes, steps, variance, clip, conf_thres, iou_thres, image_size, device):
		self.min_sizes = min_sizes
		self.steps = steps
		self.variance = variance
		self.clip = clip
		self.conf_thres = conf_thres
		self.iou_thres = iou_thres
		# self.device = select_device(device)
		# self.model = torch.load(model_path, map_location=self.device)
		self.imagesz = image_size
		self.imgsz_align = [112,112]
		devices = ort.get_available_providers()
		print(devices)
		if 'CUDAExecutionProvider' in devices:
			providers = ['CPUExecutionProvider', 'CUDAExecutionProvider']
		else:
			providers = ['CPUExecutionProvider']
		self.sess = ort.InferenceSession(model_path, providers=devices)
		priorbox = PriorBox(min_sizes=self.min_sizes, steps=self.steps, clip=self.clip, image_size=self.imagesz)
		self.priors = priorbox.forward()
		# priors = priors.to(self.device)

	def preProcess(self, img):
		h, w, _ = img.shape
		scale = np.array([w, h, w, h])

		# # Calculate widht and height and paddings
		# r_w = self.imagesz[1] / w
		# r_h = self.imagesz[0] / h
		# if r_h > r_w:
		# 	tw = self.imagesz[1]
		# 	th = int(r_w * h)
		# 	tx1 = tx2 = 0
		# 	ty1 = int((self.imagesz[0] - th) / 2)
		# 	ty2 = self.imagesz[0] - th - ty1
		# else:
		# 	tw = int(r_h * w)
		# 	th = self.imagesz[0]
		# 	tx1 = int((self.imagesz[1] - tw) / 2)
		# 	tx2 = self.imagesz[1] - tw - tx1
		# 	ty1 = ty2 = 0

		# # Resize the image with long side while maintaining ratio
		# img = cv2.resize(img, (tw,th), interpolation=cv2.INTER_AREA)
		# # Pad the short side with (128,128,128)
		# img = cv2.copyMakeBorder(
		# 	img, ty1, ty2, tx1, tx2, cv2.BORDER_CONSTANT, (128, 128, 128)
		# )
		img = cv2.resize(img, (640,640), interpolation=cv2.INTER_AREA)
		img = img.astype(np.float32)
		# HWC to CHW format:
		img -= (104, 117, 123)
		img = img.transpose(2, 0, 1)
		# CHW to NCHW format
		img = np.expand_dims(img, axis=0)
		# img = img.to(self.device)
		# scale = scale.to(self.device)
		img = np.ascontiguousarray(img)
		return [img, scale, h, w]

	def postProcess(self, input, output):
		img, scale, im_height, im_width = input
		loc, conf, landms = output
		# st_time = time.time()
		prior_data = self.priors.copy()
		prior_data = torch.from_numpy(prior_data)
		# print("----Duration PriorBox: ", time.time()-st_time)
		# stt_time = time.time()
		# st_time = time.time()
		boxes = self.decode_cpu(loc.squeeze(0), prior_data.cpu().numpy(), self.variance)
		# print("----Duration decode_cpu: ", time.time()-st_time)
		boxes = boxes * scale
		# boxes = boxes.cpu().numpy()
		scores = conf.squeeze(0)[:, 1]
		# st_time = time.time()
		landms = self.decode_landm_cpu(landms.squeeze(0), prior_data.cpu().numpy(), self.variance)
		# print("----Duration decode_landm: ", time.time()-st_time)
		scale_landms = np.array([im_width, im_height, im_width, im_height, im_width,
							   im_height, im_width, im_height, im_width, im_height])
		# scale_landms = scale_landms.cpu().numpy()
		landms = landms * scale_landms
		# print("----Duration test: ", time.time()-stt_time)

		inds = np.where(scores > self.conf_thres)[0]
		boxes = boxes[inds]
		landms = landms[inds]
		scores = scores[inds]
		order = scores.argsort()[::-1][:5000]
		boxes = boxes[order]
		landms = landms[order]
		scores = scores[order]
		dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
		# st_time = time.time()
		keep = self.py_cpu_nms(dets, self.iou_thres)
		# print("----Duration nms: ", time.time()-st_time)
		dets = dets[keep, :]
		landms = landms[keep]

		dets = dets[:750, :]
		landms = landms[:750, :]
		dets = np.concatenate((dets, landms), axis=1)
		return dets

	def py_cpu_nms(self, dets, thresh):
		x1 = dets[:, 0]
		y1 = dets[:, 1]
		x2 = dets[:, 2]
		y2 = dets[:, 3]
		scores = dets[:, 4]
		areas = (x2 - x1 + 1) * (y2 - y1 + 1)
		order = scores.argsort()[::-1]
		keep = []
		while order.size > 0:
			i = order[0]
			keep.append(i)
			xx1 = np.maximum(x1[i], x1[order[1:]])
			yy1 = np.maximum(y1[i], y1[order[1:]])
			xx2 = np.minimum(x2[i], x2[order[1:]])
			yy2 = np.minimum(y2[i], y2[order[1:]])
			w = np.maximum(0.0, xx2 - xx1 + 1)
			h = np.maximum(0.0, yy2 - yy1 + 1)
			inter = w * h
			ovr = inter / (areas[i] + areas[order[1:]] - inter)
			inds = np.where(ovr <= thresh)[0]
			order = order[inds + 1]
		return keep

	def decode(self, loc, priors, variances):
		boxes = torch.cat((priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
							priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
		boxes[:, :2] -= boxes[:, 2:] / 2
		boxes[:, 2:] += boxes[:, :2]
		return boxes

	def decode_cpu(self, loc, priors, variances):
		boxes = np.concatenate((priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
							priors[:, 2:] * np.exp(loc[:, 2:] * variances[1])), 1)
		boxes[:, :2] -= boxes[:, 2:] / 2
		boxes[:, 2:] += boxes[:, :2]
		return boxes

	def decode_landm(self, pre, priors, variances):
		landms = torch.cat((priors[:, :2] + pre[:, :2] * variances[0] * priors[:, 2:],
							priors[:, :2] + pre[:, 2:4] * variances[0] * priors[:, 2:],
							priors[:, :2] + pre[:, 4:6] * variances[0] * priors[:, 2:],
							priors[:, :2] + pre[:, 6:8] * variances[0] * priors[:, 2:],
							priors[:, :2] + pre[:, 8:10] * variances[0] * priors[:, 2:],
							), dim=1)
		return landms

	def decode_landm_cpu(self, pre, priors, variances):
		landms = np.concatenate((priors[:, :2] + pre[:, :2] * variances[0] * priors[:, 2:],
							priors[:, :2] + pre[:, 2:4] * variances[0] * priors[:, 2:],
							priors[:, :2] + pre[:, 4:6] * variances[0] * priors[:, 2:],
							priors[:, :2] + pre[:, 6:8] * variances[0] * priors[:, 2:],
							priors[:, :2] + pre[:, 8:10] * variances[0] * priors[:, 2:],
							), axis=1)
		return landms

	def inference(self, ims):
		results = []
		miss_det = []
		croped_images = []
		for i, im in enumerate(ims):
			(h, w) = im.shape[:2]
			# im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
			# st_time = time.time()
			input = self.preProcess(im)
			# print("----Duration preprocess retinaface: ", time.time()-st_time)
			img = input[0]
			# loc, conf, landms = self.model(img)
			ort_inputs = {self.sess.get_inputs()[0].name: img}
			# st_time = time.time()
			loc, conf, landms = self.sess.run(None, ort_inputs)
			# print("----Duration infer: ", time.time()-st_time)
			output = [loc, conf, landms]
			# st_time = time.time()
			dets = self.postProcess(input, output)
			# print("----Duration postprocess retinaface: ", time.time()-st_time)
			if len(dets) != 0:
				# result = dict(loc=dets[:,:4], conf=dets[:,4], landms=dets[:,5:])
				# results.append(result)
				dets_max = max(dets.tolist(), key=lambda x: (x[2]-x[0])*(x[3]-x[1]))
				dets_max = np.array(dets_max)
				dets_max[:4:2] = np.clip(dets_max[:4:2], a_min=0, a_max=w)
				dets_max[1:4:2] = np.clip(dets_max[1:4:2], a_min=0, a_max=h)
				dets_max[5::2] = np.clip(dets_max[5::2], a_min=0, a_max=w)
				dets_max[6::2] = np.clip(dets_max[6::2], a_min=0, a_max=h)
				bbox = np.array(dets_max[:4])
				area = (bbox[2]-bbox[0])*(bbox[3]-bbox[1])
				print(f"----area: {area}")
				print(f"----frame_area: {h*w*0.02}")
				if area < h*w*0.02:
					miss_det.append(i)
					continue
				result = dict(loc=dets_max[:4], conf=dets_max[4], landms=dets_max[5:])
				results.append(result)
				landmarks = np.array(dets_max[5:])
				landmarks = np.array([landmarks[0], landmarks[2], landmarks[4], landmarks[6], landmarks[8],
							landmarks[1], landmarks[3], landmarks[5], landmarks[7], landmarks[9]])
				landmarks = landmarks.reshape((2,5)).T
				# st_time = time.time()
				nimg = face_preprocess(im, bbox, landmarks, image_size=self.imgsz_align)
				# print("----Duration align: ", time.time()-st_time)
				# cv2.imwrite("aaaaa.jpg", nimg)
				croped_images.append(nimg)
			else:
				miss_det.append(i)
		return np.array(results), np.array(miss_det), np.array(croped_images)

	def preProcess_batch(self, ims):
		ims = np.array(ims)
		ims = ims.astype(np.float32)
		# HWC to CHW format:
		ims -= (104, 117, 123)
		ims = ims.transpose(0, 3, 1, 2)
		# CHW to NCHW format
		ims = np.ascontiguousarray(ims)
		return ims

	def postProcess_batch(self, ims, sizes, locs, confs, landms):
		# img, scale, im_height, im_width = input
		# loc, conf, landms = output
		# scale_boxes = np.tile(sizes, (2))
		# st_time = time.time()
		prior_data = self.priors.copy()
		prior_datas = np.tile(prior_data, (len(sizes), 1, 1))
		# prior_data = torch.from_numpy(prior_data)
		# print("----Duration PriorBox: ", time.time()-st_time)
		# stt_time = time.time()
		# st_time = time.time()
		boxes = self.decode_cpu_batch(locs, prior_datas, self.variance)
		# print("----Duration decode_cpu: ", time.time()-st_time)
		scale_boxes = np.tile(sizes, (1,boxes.shape[1],2))
		boxes = boxes * scale_boxes
		# boxes = boxes.cpu().numpy()
		scores = confs[:, :, 1]
		# st_time = time.time()
		landms = self.decode_landm_cpu_batch(landms, prior_datas, self.variance)
		# print("----Duration decode_landm: ", time.time()-st_time)
		scale_landms = np.tile(sizes, (1,landms.shape[1],5))
		landms = landms * scale_landms
		# print("----Duration test: ", time.time()-stt_time)

		results = []
		miss_det = []
		croped_images = []
		for im, sz, b, s, l in zip(ims, sizes, boxes, scores, landms):
			sz = sz[0]
			inds = np.where(s > self.conf_thres)[0]
			b = b[inds]
			l = l[inds]
			s = s[inds]
			order = s.argsort()[::-1][:5000]
			b = b[order]
			l = l[order]
			s = s[order]
			dets = np.hstack((b, s[:, np.newaxis])).astype(np.float32, copy=False)
			# st_time = time.time()
			keep = self.py_cpu_nms(dets, self.iou_thres)
			# print("----Duration nms: ", time.time()-st_time)
			dets = dets[keep, :]
			l = l[keep]

			dets = dets[:750, :]
			l = l[:750, :]
			dets = np.concatenate((dets, l), axis=1)

			# find the biggest face
			if len(dets) != 0:
				dets_max = max(dets.tolist(), key=lambda x: (x[2]-x[0])*(x[3]-x[1]))
				dets_max = np.array(dets_max)
				dets_max[:4:2] = np.clip(dets_max[:4:2], a_min=0, a_max=sz[0])
				dets_max[1:4:2] = np.clip(dets_max[1:4:2], a_min=0, a_max=sz[1])
				dets_max[5::2] = np.clip(dets_max[5::2], a_min=0, a_max=sz[0])
				dets_max[6::2] = np.clip(dets_max[6::2], a_min=0, a_max=sz[1])
				bbox = np.array(dets_max[:4])
				area = (bbox[2]-bbox[0])*(bbox[3]-bbox[1])
				# print(f"----area: {area}")
				# print(f"----frame_area: {sz[1]*sz[0]*0.02}")
				if area < sz[1]*sz[0]*0.02:
					miss_det.append(i)
					# croped_images.append(None)
					continue
				result = dict(loc=dets_max[:4], conf=dets_max[4], landms=dets_max[5:])
				results.append(result)
				landmarks = np.array(dets_max[5:])
				landmarks = np.array([landmarks[0], landmarks[2], landmarks[4], landmarks[6], landmarks[8],
							landmarks[1], landmarks[3], landmarks[5], landmarks[7], landmarks[9]])
				landmarks = landmarks.reshape((2,5)).T
				nimg = face_preprocess(im, bbox, landmarks, image_size=self.imgsz_align)
				croped_images.append(nimg)
			else:
				miss_det.append(i)
		return (np.array(results), np.array(miss_det), np.array(croped_images))

	def decode_cpu_batch(self, locs, priors, variances):
		boxes = np.concatenate((priors[:, :, :2] + locs[:, :, :2] * variances[0] * priors[:, :, 2:],
							priors[:, :, 2:] * np.exp(locs[:, :, 2:] * variances[1])), 2)
		boxes[:, :, :2] -= boxes[:, :, 2:] / 2
		boxes[:, :, 2:] += boxes[:, :, :2]
		return boxes

	def decode_landm_cpu_batch(self, pre, priors, variances):
		landms = np.concatenate((priors[:, :, :2] + pre[:, :, :2] * variances[0] * priors[:, :, 2:],
							priors[:, :, :2] + pre[:, :, 2:4] * variances[0] * priors[:, :, 2:],
							priors[:, :, :2] + pre[:, :, 4:6] * variances[0] * priors[:, :, 2:],
							priors[:, :, :2] + pre[:, :, 6:8] * variances[0] * priors[:, :, 2:],
							priors[:, :, :2] + pre[:, :, 8:10] * variances[0] * priors[:, :, 2:],
							), axis=2)
		return landms

	def inference_batch(self, ims):
		inputs = []
		sizes = []
		for i, im in enumerate(ims):
			(h, w) = im.shape[:2]
			img = cv2.resize(im, (640,640), interpolation=cv2.INTER_AREA)
			inputs.append(img)
			sizes.append([[w,h]])

		inputs = self.preProcess_batch(inputs)
		ort_inputs = {self.sess.get_inputs()[0].name: inputs}
		locs, confs, landms = self.sess.run(None, ort_inputs)
		final_result = self.postProcess_batch(ims, sizes, locs, confs, landms)
		return final_result

	def render(self, ims):
		im_preds = []
		for i, im in enumerate(ims):
			# im = np.array(im.convert('RGB'))
			im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
			input = self.preProcess(im)
			img = input[0]
			loc, conf, landms = self.model(img)
			output = [loc, conf, landms]
			dets = self.postProcess(input, output)
			for det in dets:
				cv2.rectangle(im, det[:2].astype(int), det[2:4].astype(int), (255, 0, 0), 2)
			# cv2.imwrite("dfsdf.jpg", im)
			im_preds.append(im)
		return im_preds

if __name__=="__main__":
	CONFIG_FACEDET = {
		"model_path": f"./weights/detectFace_model_op16.onnx",
		"min_sizes": [[16,32], [64,128], [256,512]],
		"steps": [8, 16, 32],
		"variance": [0.1, 0.2],
		"clip": False,
		"conf_thres": 0.75,
		"iou_thres": 0.25,
		"image_size": [640,640],
		"device": "cpu",
	}

	facedet = RetinanetRunnable(**CONFIG_FACEDET)
	img = cv2.imread("test.jpg")
	ims = [img,img,img]
	results, miss_det, croped_images = facedet.inference_batch(ims)
	print(results)
	print(croped_images.shape)