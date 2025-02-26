import os, sys
import numpy as np
import tritonclient.grpc.aio as grpcclient

def get_triton_client(ip_address="127.0.0.1:8001"):
	try:
		triton_client = grpcclient.InferenceServerClient(
			url=ip_address
		)
	except Exception as e:
		print("channel creation failed: " + str(e))
		sys.exit()

	return triton_client

def get_io_retinaface(imgs):
	if len(imgs.shape) < 4:
		imgs = np.expand_dims(imgs, axis=0)
	# Infer
	inputs = []
	outputs = []
	inputs.append(grpcclient.InferInput("input_image", imgs.shape, "UINT8"))

	# Initialize the data
	inputs[0].set_data_from_numpy(imgs)

	outputs.append(grpcclient.InferRequestedOutput("croped_image"))
	outputs.append(grpcclient.InferRequestedOutput("preprocessed_image_info"))

	return inputs, outputs

def get_io_ghostface(imgs):
	if len(imgs.shape) < 4:
		imgs = np.expand_dims(imgs, axis=0)
	# Infer
	inputs = []
	outputs = []
	inputs.append(grpcclient.InferInput("input_image", imgs.shape, "UINT8"))

	# Initialize the data
	inputs[0].set_data_from_numpy(imgs)

	outputs.append(grpcclient.InferRequestedOutput("feature_norm"))

	return inputs, outputs


def get_io_compareface(ft1, ft2):
	# Infer
	inputs = []
	outputs = []
	inputs.append(grpcclient.InferInput("compare_face_feature1", ft1.shape, "FP16"))
	inputs.append(grpcclient.InferInput("compare_face_feature2", ft2.shape, "FP16"))

	# Initialize the data
	inputs[0].set_data_from_numpy(ft1)
	inputs[1].set_data_from_numpy(ft2)

	outputs.append(grpcclient.InferRequestedOutput("similarity"))
	outputs.append(grpcclient.InferRequestedOutput("similarity_sort_idx"))

	return inputs, outputs
