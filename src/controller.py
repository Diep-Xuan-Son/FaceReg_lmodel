import asyncio
import uvicorn
# import redis
import threading
from io import BytesIO
import multiprocessing
from string import ascii_letters, digits, punctuation

from app import *

heart_beat_thread = threading.Thread(target=delete_file_cronj, args=(PATH_IMG_FAIL, 25200), daemon=True)
heart_beat_thread.start()

@app.post("/api/registerFace")
async def registerFace(params: Person = Depends(Person.as_form), images: List[UploadFile] = File(...)):
	try:
		code = params.code
		# print(code)
		special_letters = set(code).difference(ascii_letters + digits + punctuation)
		if special_letters:
			return {"success": False, "error_code": 8010, "error": "There are some special letters in user code!"}
		if redisClient.hexists("FaceInfor1", code):
			return {"success": False, "error_code": 8004, "error": "This user has been registered!"}

		path_avatar = f"{IMG_AVATAR}/{code}/face_1.jpg"
		path_code = os.path.join(PATH_IMG_AVATAR, code)
		if os.path.exists(path_code):
			shutil.rmtree(path_code)
		os.mkdir(path_code)

		name = params.name
		birthday = params.birthday
		imgs = []
		img_infor = []

		for i, image in enumerate(images):
			image_byte = await image.read()
			nparr = np.fromstring(image_byte, np.uint8)
			img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
			img_infor.append(img.shape[:2])
			cv2.imwrite(f'{path_code}/face_{i+1}.jpg', img)
			# img = cv2.resize(img, (640,640), interpolation=cv2.INTER_AREA)
			imgs.append(img)
		imgs = np.array(imgs)
		img_infor = np.array(img_infor)
		#---------------------------face det-------------------------
		results = FACEDET.inference(imgs)
		dets, miss_det, croped_image = results
		if len(croped_image)==0:
			return {"success": False, "error_code": 8001, "error": "Don't find any face"}
		# print(croped_image.shape)
		# cv2.imwrite("sadas.jpg", croped_image[0])
		#////////////////////////////////////////////////////////////

		#---------------------------face reg-------------------------
		feature = GHOSTFACE.get_feature_without_det(croped_image)
		feature = np.array(feature, dtype=np.float16)
		LOGGER_APP.info(f"------feature: {feature.shape}")

		# dt_person_information = {code: f"{code},./{name},./{birthday}"}
		# dt_person_feature = {code: feature.tobytes()}
		redisClient.hset("FaceInfor1", code, f"{code}@@@{name}@@@{birthday}@@@{path_avatar}")
		redisClient.hset("FaceFeature1", code, feature.tobytes())

		return {"success": True}
	except Exception as e:
		tb_str = traceback.format_exc()
		LOGGER_APP.error(f"Traceback: {tb_str}")
		return {"success": False, "error_code": 8008, "error": str(e)}

@app.post("/api/deleteUser")
def deleteUser(codes: List[str] = ["001099008839"]):
	LOGGER_APP.info(f"----codes: {codes}")
	try:
		codes_noregister = []
		for code in codes:
			if not redisClient.hexists("FaceInfor1", code) or not redisClient.hexists("FaceFeature1", code):
				codes_noregister.append(code)
		if len(codes_noregister)>0:
			return {"success": False, "error_code": 8006, "error": f"User {tuple(codes_noregister)} has not been registered!"}

		redisClient.hdel("FaceInfor1", *codes)
		redisClient.hdel("FaceFeature1", *codes)

		for code in codes:
			path_code = os.path.join(PATH_IMG_AVATAR, code)
			if os.path.exists(path_code):
				shutil.rmtree(path_code)

		return {"success": True}
	except Exception as e:
		tb_str = traceback.format_exc()
		LOGGER_APP.error(f"Traceback: {tb_str}")
		return {"success": False, "error_code": 8008, "error": str(e)}

@app.post("/api/deleteAllUser")
def deleteAllUser():
	try:
		redisClient.delete("FaceInfor1")
		redisClient.delete("FaceFeature1")
		if os.path.exists(PATH_IMG_AVATAR):
			shutil.rmtree(PATH_IMG_AVATAR)
			os.mkdir(PATH_IMG_AVATAR)
		return {"success": True}
	except Exception as e:
		tb_str = traceback.format_exc()
		LOGGER_APP.error(f"Traceback: {tb_str}")
		return {"success": False, "error_code": 8008, "error": str(e)}

@app.post("/api/getInformationUser")
def getInformationUser(codes: List[str] = []):
	try:
		infor_persons = {}
		LOGGER_APP.info(f"----codes: {codes}")
		if len(codes)==0:
			key_infor_persons = redisClient.hkeys("FaceInfor1")
			if len(key_infor_persons)==0:
				return {"success": True, "information": infor_persons}
			key_infor_persons = b'-;'.join(key_infor_persons).decode('utf-8').split("-;")
			val_infor_persons = redisClient.hvals("FaceInfor1")
			val_infor_persons = np.array(b'@@@'.join(val_infor_persons).decode('utf-8').split("@@@")).reshape(-1,4)	# shape (-1,3) for 3 field: code, name, birthday
			infor_persons = dict(zip(key_infor_persons, val_infor_persons.tolist()))
		else:
			for code in codes:
				LOGGER_APP.info(redisClient.hexists("FaceInfor1", code))
				if not redisClient.hexists("FaceInfor1", code):
					infor_persons[code] = "No register"
					continue
				infor_person = redisClient.hget("FaceInfor1", code)
				infor_person = infor_person.decode("utf-8").split("@@@")
				infor_persons[code] = {"id": infor_person[0], \
										"name": infor_person[1], \
										"birthday": infor_person[2], \
										"avatar": infor_person[3]
										}
		return {"success": True, "information": infor_persons}
	except Exception as e:
		tb_str = traceback.format_exc()
		LOGGER_APP.error(f"Traceback: {tb_str}")
		return {"success": False, "error_code": 8008, "error": str(e)}

@app.post("/api/searchUser")
async def searchUser(image: UploadFile = File(...)):
	try:
		id_faces = redisClient.hkeys("FaceFeature1")
		if len(id_faces) == 0:
			return {"success": False, "error_code": 8000, "error": "Don't have any registered user"}
		image_byte = await image.read()
		nparr = np.fromstring(image_byte, np.uint8)
		img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
		#----------------
		# img_list = os.listdir("./image_error")
		# cv2.imwrite(f"aaa.jpg", img)
		#------------------
		t_det = time.time()
		#---------------------------face det-------------------------
		results = FACEDET.inference([img])
		dets, miss_det, croped_image = results
		if len(croped_image)==0:
			return {"success": False, "error_code": 8001, "error": "Don't find any face"}
			
		box = dets[0]["loc"]
		# print((box[2]-box[0])*(box[3]-box[1]))
		area_img = img.shape[0]*img.shape[1]
		w_crop = (box[2]-box[0])
		h_crop = (box[3]-box[1])
		# if not area_img*0.1<w_crop*h_crop<area_img*0.3:
		# 	return {"success": False, "error_code": 8001, "error": "Face size is not true"}
		#---------------spoofing--------------
		box_expand = np.array([max(box[0]-w_crop,0), max(box[1]-h_crop,0), min(box[2]+w_crop, img.shape[1]), min(box[3]+h_crop, img.shape[0])], dtype=int)
		result = SPOOFINGDET.inference([img[box_expand[1]:box_expand[3], box_expand[0]:box_expand[2]]])[0]
		LOGGER_APP.info(f"---------result_spoofing: {result}")
		# cv2.imwrite(f"aaa.jpg", img[box_expand[1]:box_expand[3], box_expand[0]:box_expand[2]])
		if result[1] > 0.78:
			#img_list = os.listdir(f"{PATH_IMG_SPOOFING}")
			#cv2.imwrite(f"{PATH_IMG_SPOOFING}/{len(img_list)}.jpg", img[box_expand[1]:box_expand[3], box_expand[0]:box_expand[2]])
			return {"success": False, "error_code": 8002, "error": "Fake face image"}
		#//////////////////////////////////////
		#////////////////////////////////////////////////////////////
		LOGGER_APP.info(f"------Duration det: {time.time()-t_det}")

		t_reg = time.time()
		#---------------------------face reg-------------------------
		feature = GHOSTFACE.get_feature_without_det(croped_image)
		feature = np.array(feature, dtype=np.float16)
		LOGGER_APP.info(f"------ghostface_feature: {feature.shape}")

		# in_ghostface, out_ghostface = get_io_ghostface(croped_image)
		# results = await tritonClient.infer(model_name="ghost_face_nodet_ensemble", inputs=in_ghostface, outputs=out_ghostface)
		# feature = results.as_numpy("feature_norm")
		# feature = feature.astype(np.float16)
		#////////////////////////////////////////////////////////////
		LOGGER_APP.info(f"------Duration reg: {time.time()-t_reg}")

		#---------------------------compare fail face---------------------
		ft_faces = np.array(redisClient.hvals("FaceFeatureFail2"))
		if len(ft_faces)==0:
			codes_fail = []
		else:
			feature_truth = np.frombuffer(ft_faces, dtype=np.float16).reshape(len(ft_faces), 512)
			# print(feature_truth.shape)
			# print(feature.shape)
			dist = np.linalg.norm(feature - feature_truth, axis=1)
			similarity = (np.tanh((1.23132175 - dist) * 6.602259425) + 1) / 2

			ft_faces_idx = np.char.decode(redisClient.hkeys("FaceFeatureFail2"), encoding="utf-8").astype(int)
			code_list = np.char.decode(redisClient.hvals("FaceListCodeFail2"), encoding="utf-8")

			codes, idx = np.unique(code_list, return_inverse=True)	# get unique code with corresponding index 
			ft_faces_idx_sort = np.argsort(ft_faces_idx, axis=0)	# get index of sorted value
			similarity = similarity[ft_faces_idx_sort]				# value with sorted key
			similarity_average = np.bincount(idx.flatten(), weights = similarity.flatten())/np.bincount(idx.flatten())	# calculate average with the same unique index 
			# print(similarity_average)
			rand = np.random.random(similarity_average.size)
			idx_sorted = np.lexsort((rand,similarity_average))[::-1] #sort random index by similarity_average
			# print(idx_sorted)
			similaritys = similarity_average[idx_sorted]
			codes_fail = codes[idx_sorted][similaritys>0.8]
		#/////////////////////////////////////////////////////////////////

		t_db = time.time()
		ft_faces = np.array(redisClient.hvals("FaceFeature1"))
		feature_truth = np.frombuffer(ft_faces, dtype=np.float16).reshape(len(ft_faces), -1, 512)
		LOGGER_APP.info(f"------Duration db: {time.time()-t_db}")

		t_comp = time.time()
		#---------------------------compare face----------------------
		similarity, similarity_sort_idx = GHOSTFACE.compare_face_1_n_n(feature, feature_truth)
		# dist = np.linalg.norm(feature[:,::1] - feature_truth[:,:,::], axis=2)
		# similarity = (np.tanh((1.23132175 - dist) * 6.602259425) + 1) / 2
		# similarity = np.mean(similarity, axis=1)
		# rand = np.random.random(similarity.size)
		# similarity_sort_idx = np.lexsort((rand,similarity))[::-1]
		similarity_sort_idx_best = similarity_sort_idx[0]
		similarity_best = similarity[similarity_sort_idx_best]
		LOGGER_APP.info(f"---------similarity_best: {similarity_best}")

		# infor_face = None
		# if similarity_best > 0.70:
		# 	id_faces_best = id_faces[similarity_sort_idx_best]
		# 	infor_face = redisClient.hget("FaceInfor1", id_faces_best)

		infor_face = None
		similaritys = similarity[similarity_sort_idx]
		id_faces = np.array(id_faces, dtype=np.str_)
		codes = id_faces[similarity_sort_idx][similaritys>0.75]
		LOGGER_APP.info(f"-------codes_fail: {codes_fail}")
		LOGGER_APP.info(f"-------codes: {codes}")
		LOGGER_APP.info(f"-------similaritys: { similaritys[similaritys>0.75]}")
		for i, code in enumerate(codes):
			if code not in codes_fail:
				similarity_best = similarity[similarity_sort_idx[i]]
				infor_face = redisClient.hget("FaceInfor1", code)
				break
		#/////////////////////////////////////////////////////////////
		LOGGER_APP.info("------Duration compare: {time.time()-t_comp}")
		if infor_face is None:
			name_fail_img = datetime.now().strftime('%Y-%m-%d_%H-%M')
			cv2.imwrite(f'{PATH_IMG_FAIL}/{name_fail_img}.jpg', img)
			return {"success": False, "error_code": 8003, "error": "Don't find any user"}
		LOGGER_APP.info(f"----infor_face: {infor_face}")
		infor_face = infor_face.decode("utf-8").split("@@@")
		return {"success": True, "information": {"code": infor_face[0], "name": infor_face[1], "birthday": infor_face[2], "avatar": infor_face[3], "similarity": float(similarity_best)}}
	except Exception as e:
		tb_str = traceback.format_exc()
		LOGGER_APP.error(f"Traceback: {tb_str}")
		return {"success": False, "error_code": 8008, "error": str(e)}

@app.post("/api/spoofingCheck")
async def spoofingCheck(image: UploadFile = File(...)):
	try:
		image_byte = await image.read()
		nparr = np.fromstring(image_byte, np.uint8)
		img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
		#---------------------------face det-------------------------
		results = FACEDET.inference([img])
		dets, miss_det, croped_image = results
		if len(croped_image)==0:
			return {"success": False, "error_code": 8001, "error": "Don't find any face"}
		#---------------spoofing--------------
		result = SPOOFINGDET.inference([img])[0]
		LOGGER_APP.info(f"---------result_spoofing: {result}")
		if result[1] > 0.85:
			# img_list = os.listdir(f"{PATH_IMG_SPOOFING}")
			# cv2.imwrite(f"{PATH_IMG_SPOOFING}/{len(img_list)}.jpg", img)
			return {"success": False, "error_code": 8002, "error": "Fake face image"}
		return {"success": True}
		#//////////////////////////////////////
	except Exception as e:
		tb_str = traceback.format_exc()
		LOGGER_APP.error(f"Traceback: {tb_str}")
		return {"success": False, "error": str(e)}

@app.post("/api/compareFace")
async def compareFace(image_face: UploadFile = File(...), image_identification: UploadFile = File(...)):
	try:
		image_byte = await image_face.read()
		nparr = np.fromstring(image_byte, np.uint8)
		img_face = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

		image_byte = await image_identification.read()
		nparr = np.fromstring(image_byte, np.uint8)
		img_id = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
		st_time = time.time()
		t_det = time.time()
		#---------------------------face det-------------------------
		results = await asyncio.gather(*[
				asyncio.to_thread(FACEDET.inference, [img_face]),
				asyncio.to_thread(FACEDET.inference, [img_id])
			])

		dets_face, miss_det, croped_image_face = results[0]
		dets_id, miss_det, croped_image_id = results[1]
		if len(dets_face) == 0:
			return {"success": False, "error_code": 8001, "error": "Don't find any face in identification photo"}

		box_face = dets_face[0]["loc"]
		area_img = img_face.shape[0]*img_face.shape[1]
		w_crop = (box_face[2]-box_face[0])
		h_crop = (box_face[3]-box_face[1])
		#---------------spoofing--------------
		box_expand = np.array([max(box_face[0]-w_crop,0), max(box_face[1]-h_crop,0), min(box_face[2]+w_crop, img_face.shape[1]), min(box_face[3]+h_crop, img_face.shape[0])], dtype=int)
		result = SPOOFINGDET.inference([img_face[box_expand[1]:box_expand[3], box_expand[0]:box_expand[2]]])[0]
		LOGGER_APP.info(f"---------result_spoofing: {result}")
		if result[1] > 0.85:
			# img_list = os.listdir("./image_test")
			# cv2.imwrite(f"./image_test/{len(img_list)}.jpg", img_spoofing)
			return {"success": False, "error_code": 8002, "error": "Fake face image"+quality_mes}
		#//////////////////////////////////////
		#////////////////////////////////////////////////////////////
		LOGGER_APP.info(f"------Duration det: {time.time()-t_det}")
		
		t_reg = time.time()
		#---------------------------face reg-------------------------
		features = await asyncio.gather(*[
				asyncio.to_thread(GHOSTFACE.get_feature_without_det, croped_image_face),
				asyncio.to_thread(GHOSTFACE.get_feature_without_det, croped_image_id)
			])
		feature_face = features[0]
		feature_face = np.array(feature_face, dtype=np.float16)
		feature_id = features[1]
		feature_id = np.array(feature_id, dtype=np.float16)
		#////////////////////////////////////////////////////////////
		LOGGER_APP.info(f"----Duration reg: {time.time()-t_reg}")

		t_comp = time.time()
		#---------------------------compare face----------------------
		# print(feature_face.shape)
		# print(feature_id.shape)
		similarity, similarity_sort_idx = GHOSTFACE.compare_face_1_n_1(feature_face, feature_id)
		similarity = similarity[0]

		LOGGER_APP.info(f"----similarity: {similarity}")
		LOGGER_APP.info(f"----Total time duration: {time.time()-st_time}")

		if similarity < 0.8499:
			name_fail_img = datetime.now().strftime('%Y-%m-%d_%H-%M')
			cv2.imwrite(f'{PATH_IMG_FAIL}/{name_fail_img+"_0"}.jpg', img_face)
			cv2.imwrite(f'{PATH_IMG_FAIL}/{name_fail_img+"_1"}.jpg', img_id)
			return {"success": False, "error_code": 8003, "error": "Face photo and identification photo is not similar"}
		return {"success": True, "similarity": float(similarity)}
		#/////////////////////////////////////////////////////////////
	except Exception as e:
		tb_str = traceback.format_exc()
		LOGGER_APP.error(f"Traceback: {tb_str}")
		return {"success": False, "error_code": 8008, "error": str(e)}

@app.post('/healthcheck')
async def health_check():
	return { 'success': True, 'message': "healthy" }

@app.post("/api/registerFacev2")
async def registerFacev2(params: Person = Depends(Person.as_form), images: List[UploadFile] = File(...)):
	try:
		code = params.code
		LOGGER_APP.info(f"----code: {code}")
		special_letters = set(code).difference(ascii_letters + digits + punctuation)
		if special_letters:
			return {"success": False, "error_code": 8010, "error": "There are some special letters in user code!"}
		# if redisClient.hexists("FaceInfor2", code):
		# 	return {"success": False, "error_code": 8004, "error": "This user has been registered!"}

		path_avatar = f"{IMG_AVATAR}/{code}/face_0.jpg"
		path_code = os.path.join(PATH_IMG_AVATAR, code)
		os.makedirs(path_code, exist_ok=True)

		name = params.name
		birthday = params.birthday
		imgs = []
		img_infor = []

		for i, image in enumerate(images):
			image_byte = await image.read()
			nparr = np.fromstring(image_byte, np.uint8)
			img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
			img_infor.append(img.shape[:2])
			# cv2.imwrite(f'{path_code}/face_{i}.jpg', img)
			# img = cv2.resize(img, (640,640), interpolation=cv2.INTER_AREA)
			imgs.append(img)
		imgs = np.array(imgs)
		img_infor = np.array(img_infor)
		#---------------------------face det-------------------------
		results = FACEDET.inference(imgs)
		dets, miss_det, croped_image = results
		if len(croped_image)==0:
			return {"success": False, "error_code": 8001, "error": "Don't find any face"}
		# print(croped_image.shape)
		# cv2.imwrite("sadas.jpg", croped_image[0])
		#////////////////////////////////////////////////////////////

		#---------------------------face reg-------------------------
		feature = GHOSTFACE.get_feature_without_det(croped_image)
		feature = np.array(feature, dtype=np.float16)
		# print(feature.shape)

		id_faces = redisClient.hkeys("FaceListCode2")
		LOGGER_APP.info(f"----num id_face: {len(id_faces)}")
		if len(id_faces)!=0:
			id_faces = int(id_faces[-1]) + 1
		else:
			id_faces = 0

		redisClient.hset("FaceInfor2", code, f"{code}@@@{name}@@@{birthday}@@@{path_avatar}")
		for i, ft in enumerate(feature):
			num_face = len(os.listdir(path_code))
			cv2.imwrite(f'{path_code}/face_{num_face}.jpg', imgs[i])
			redisClient.hset("FaceFeature2", f"{id_faces+i}", ft.tobytes())
			redisClient.hset("FaceListCode2", f"{id_faces+i}", f"{code}")

		return {"success": True}
	except Exception as e:
		tb_str = traceback.format_exc()
		LOGGER_APP.error(f"Traceback: {tb_str}")
		return {"success": False, "error_code": 8008, "error": str(e)}

@app.post("/api/deleteAllUserv2")
def deleteAllUserv2():
	try:
		redisClient.delete("FaceInfor2")
		redisClient.delete("FaceFeature2")
		redisClient.delete("FaceListCode2")
		if os.path.exists(PATH_IMG_AVATAR):
			shutil.rmtree(PATH_IMG_AVATAR)
			os.mkdir(PATH_IMG_AVATAR)
		return {"success": True}
	except Exception as e:
		tb_str = traceback.format_exc()
		LOGGER_APP.error(f"Traceback: {tb_str}")
		return {"success": False, "error_code": 8008, "error": str(e)}

@app.post("/api/deleteUserv2")
def deleteUserv2(codes: List[str] = ["001099008839"]):
	LOGGER_APP.info(f"----codes: {codes}")
	try:
		codes_noregister = []
		for code in codes:
			if not redisClient.hexists("FaceInfor2", code):
				codes_noregister.append(code)
		if len(codes_noregister)>0:
			return {"success": False, "error_code": 8006, "error": f"User {tuple(codes_noregister)} has not been registered!"}

		id_faces = np.char.decode(redisClient.hkeys("FaceListCode2"), encoding="utf-8")
		code_list = np.char.decode(redisClient.hvals("FaceListCode2"), encoding="utf-8")
		# print(code_list)
		# print(id_faces)
		code_idx = np.where(codes[0] == code_list)
		redisClient.hdel("FaceFeature2", *id_faces[code_idx[0].tolist()])
		redisClient.hdel("FaceListCode2", *id_faces[code_idx[0].tolist()])
		redisClient.hdel("FaceInfor2", *codes)

		for code in codes:
			path_code = os.path.join(PATH_IMG_AVATAR, code)
			if os.path.exists(path_code):
				shutil.rmtree(path_code)

		return {"success": True}
	except Exception as e:
		tb_str = traceback.format_exc()
		LOGGER_APP.error(f"Traceback: {tb_str}")
		return {"success": False, "error_code": 8008, "error": str(e)}

@app.post("/api/getInformationUserv2")
def getInformationUserv2(codes: List[str] = []):
	try:
		infor_persons = {}
		LOGGER_APP.info(f"----codes: {codes}")
		if len(codes)==0:
			key_infor_persons = redisClient.hkeys("FaceInfor2")
			if len(key_infor_persons)==0:
				return {"success": True, "information": infor_persons}
			key_infor_persons = b'-;'.join(key_infor_persons).decode('utf-8').split("-;")
			val_infor_persons = redisClient.hvals("FaceInfor2")
			val_infor_persons = np.array(b'@@@'.join(val_infor_persons).decode('utf-8').split("@@@")).reshape(-1,4)	# shape (-1,3) for 3 field: code, name, birthday
			infor_persons = dict(zip(key_infor_persons, val_infor_persons.tolist()))
		else:
			for code in codes:
				LOGGER_APP.info(redisClient.hexists("FaceInfor2", code))
				if not redisClient.hexists("FaceInfor2", code):
					infor_persons[code] = "No register"
					continue
				infor_person = redisClient.hget("FaceInfor2", code)
				infor_person = infor_person.decode("utf-8").split("@@@")
				infor_persons[code] = {"id": infor_person[0], \
										"name": infor_person[1], \
										"birthday": infor_person[2], \
										"avatar": infor_person[3]
										}
		return {"success": True, "information": infor_persons}
	except Exception as e:
		tb_str = traceback.format_exc()
		LOGGER_APP.error(f"Traceback: {tb_str}")
		return {"success": False, "error_code": 8008, "error": str(e)}

@app.post("/api/searchUserv2")
async def searchUserv2(image: UploadFile = File(...)):
	try: 
		id_faces = redisClient.hkeys("FaceInfor2")
		if len(id_faces) == 0:
			return {"success": False, "error_code": 8000, "error": "Don't have any registered user"}
		image_byte = await image.read()
		nparr = np.fromstring(image_byte, np.uint8)
		img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
		t_det = time.time()
		#---------------------------face det-------------------------
		results = FACEDET.inference([img])
		dets, miss_det, croped_image = results
		if len(croped_image)==0:
			return {"success": False, "error_code": 8001, "error": "Don't find any face"}

		box = dets[0]["loc"]
		# print((box[2]-box[0])*(box[3]-box[1]))
		area_img = img.shape[0]*img.shape[1]
		w_crop = (box[2]-box[0])
		h_crop = (box[3]-box[1])
		# if not area_img*0.15<w_crop*h_crop<area_img*0.3:
		# 	return {"success": False, "error_code": 8009, "error": "Face size is not true"}
		#---------------spoofing--------------
		box_expand = np.array([max(box[0]-w_crop,0), max(box[1]-h_crop,0), min(box[2]+w_crop, img.shape[1]), min(box[3]+h_crop, img.shape[0])], dtype=int)
		result = SPOOFINGDET.inference([img[box_expand[1]:box_expand[3], box_expand[0]:box_expand[2]]])[0]
		# result = SPOOFINGDET.inference([img])[0]
		LOGGER_APP.info(f"---------result_spoofing: {result}")
		if result[1] > 0.78:
			# img_list = os.listdir(f"{PATH_IMG_SPOOFING}")
			# cv2.imwrite(f"{PATH_IMG_SPOOFING}/{len(img_list)}.jpg", img[box_expand[1]:box_expand[3], box_expand[0]:box_expand[2]])
			return {"success": False, "error_code": 8002, "error": "Fake face image"}
		#//////////////////////////////////////
		#////////////////////////////////////////////////////////////
		LOGGER_APP.info(f"------Duration det: {time.time()-t_det}")

		t_reg = time.time()
		#---------------------------face reg-------------------------
		feature = GHOSTFACE.get_feature_without_det(croped_image)
		feature = np.array(feature, dtype=np.float16)
		#////////////////////////////////////////////////////////////
		LOGGER_APP.info(f"------Duration reg: {time.time()-t_reg}")

		t_db = time.time()
		ft_faces = np.array(redisClient.hvals("FaceFeature2"))
		feature_truth = np.frombuffer(ft_faces, dtype=np.float16).reshape(len(ft_faces), 512)
		LOGGER_APP.info(f"------Duration db: {time.time()-t_db}")

		t_comp = time.time()
		#---------------------------compare face----------------------
		dist = np.linalg.norm(feature - feature_truth, axis=1)
		similarity = (np.tanh((1.23132175 - dist) * 6.602259425) + 1) / 2
		# print(similarity)

		ft_faces_idx = np.char.decode(redisClient.hkeys("FaceFeature2"), encoding="utf-8").astype(int)
		code_list = np.char.decode(redisClient.hvals("FaceListCode2"), encoding="utf-8")

		codes, idx = np.unique(code_list, return_inverse=True)	# get unique code with corresponding index 
		ft_faces_idx_sort = np.argsort(ft_faces_idx, axis=0)	# get index of sorted value
		similarity = similarity[ft_faces_idx_sort]				# value with sorted key
		similarity_average = np.bincount(idx.flatten(), weights = similarity.flatten())/np.bincount(idx.flatten())	# calculate average with the same unique index 
		# print(similarity_average)
		rand = np.random.random(similarity_average.size)
		idx_sorted = np.lexsort((rand,similarity_average))[::-1] #sort random index by similarity_average
		similarity_best = similarity_average[idx_sorted[0]]
		LOGGER_APP.info(f"---------similarity_best: {similarity_best}")

		infor_face = None
		if similarity_best > 0.75:
			code = codes[idx_sorted[0]]
			infor_face = redisClient.hget("FaceInfor2", code)
			#save image to train
			path_user = f"{PATH_IMG_REC}/{code}"
			if not os.path.exists(path_user):
				os.makedirs(path_user, exist_ok=True)
			img_list = os.listdir(f"{path_user}")
			box = box.astype(int)
			cv2.imwrite(f"{path_user}/{len(img_list)}.jpg", img[box[1]:box[3], box[0]:box[2]])
		#/////////////////////////////////////////////////////////////
		LOGGER_APP.info(f"------Duration compare: {time.time()-t_comp}")
		if infor_face is None:
			return {"success": False, "error_code": 8003, "error": "Don't find any user"}
		LOGGER_APP.info(f"----infor_face; {infor_face}")
		infor_face = infor_face.decode("utf-8").split("@@@")
		return {"success": True, "information": {"code": infor_face[0], "name": infor_face[1], "birthday": infor_face[2], "avatar": infor_face[3], "similarity": float(similarity_best)}}
	except Exception as e:
		tb_str = traceback.format_exc()
		LOGGER_APP.error(f"Traceback: {tb_str}")
		return {"success": False, "error_code": 8008, "error": str(e)}

@app.post("/api/checkFailFacev2")
async def checkFailFacev2(params: Person = Depends(Person.as_form), images: List[UploadFile] = File(...)):
	try:
		code = params.code
		LOGGER_APP.info(f"----code: {code}")

		name = params.name
		birthday = params.birthday
		imgs = []

		num_img = len(os.listdir(PATH_IMG_FAIL))
		for i, image in enumerate(images):
			image_byte = await image.read()
			nparr = np.fromstring(image_byte, np.uint8)
			img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
			cv2.imwrite(f'{PATH_IMG_FAIL}/{code}_{name}_{num_img+i}.jpg', img)
			img = cv2.resize(img, (640,640), interpolation=cv2.INTER_AREA)
			imgs.append(img)
		imgs = np.array(imgs)
		#---------------------------face det-------------------------
		in_retinaface, out_retinaface = get_io_retinaface(imgs)
		results = await tritonClient.infer(model_name="detection_retinaface_ensemble", inputs=in_retinaface)
		croped_image = results.as_numpy("croped_image")
		num_object = results.as_numpy("num_obj").squeeze(1)

		if len(croped_image)==0:
			return {"success": True}
		#////////////////////////////////////////////////////////////

		#---------------------------face reg-------------------------
		in_ghostface, out_ghostface = get_io_ghostface(croped_image)
		results = await tritonClient.infer(model_name="ghost_face_nodet_ensemble", inputs=in_ghostface, outputs=out_ghostface)
		feature = results.as_numpy("feature_norm")
		feature = feature.astype(np.float16)

		miss_det = (np.where(num_object<1)[0]).tolist()
		[imgs.pop(idx) for idx in reversed(sorted(miss_det))]

		id_faces = redisClient.hkeys("FaceListCodeFail2")
		if len(id_faces)!=0:
			id_faces = int(id_faces[-1]) + 1
		else:
			id_faces = 0

		for i, ft in enumerate(feature):
			redisClient.hset("FaceFeatureFail2", f"{id_faces+i}", ft.tobytes())
			redisClient.hset("FaceListCodeFail2", f"{id_faces+i}", f"{code}")

		return {"success": True}
	except Exception as e:
		tb_str = traceback.format_exc()
		LOGGER_APP.error(f"Traceback: {tb_str}")
		return {"success": False, "error_code": 8008, "error": str(e)}

# @app.post("/api/getFailFacev2")
# async def getFailFacev2(image: UploadFile = File(...)):
# 	image_byte = await image.read()
# 	nparr = np.fromstring(image_byte, np.uint8)
# 	img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
# 	#---------------------------face det-------------------------
# 	in_retinaface, out_retinaface = get_io_retinaface(img)
# 	results = await tritonClient.infer(model_name="detection_retinaface_ensemble", inputs=in_retinaface)
# 	croped_image = results.as_numpy("croped_image")
# 	if len(croped_image)==0:
# 		return {"success": False, "error_code": 8001, "error": "Don't find any face"}

# 	#---------------------------face reg-------------------------
# 	in_ghostface, out_ghostface = get_io_ghostface(croped_image)
# 	results = await tritonClient.infer(model_name="recognize_face_nodet_ensemble", inputs=in_ghostface, outputs=out_ghostface)
# 	feature = results.as_numpy("feature_norm")
# 	feature = feature.astype(np.float16)
# 	#////////////////////////////////////////////////////////////

# 	ft_faces = np.array(redisClient.hvals("FaceFeatureFail2"))
# 	feature_truth = np.frombuffer(ft_faces, dtype=np.float16).reshape(len(ft_faces), 512)

# 	print(feature_truth.shape)
# 	print(feature.shape)
# 	dist = np.linalg.norm(feature - feature_truth, axis=1)
# 	similarity = (np.tanh((1.23132175 - dist) * 6.602259425) + 1) / 2

# 	ft_faces_idx = np.char.decode(redisClient.hkeys("FaceFeatureFail2"), encoding="utf-8").astype(int)
# 	code_list = np.char.decode(redisClient.hvals("FaceListCodeFail2"), encoding="utf-8")

# 	codes, idx = np.unique(code_list, return_inverse=True)	# get unique code with corresponding index 
# 	ft_faces_idx_sort = np.argsort(ft_faces_idx, axis=0)	# get index of sorted value
# 	similarity = similarity[ft_faces_idx_sort]				# value with sorted key
# 	similarity_average = np.bincount(idx.flatten(), weights = similarity.flatten())/np.bincount(idx.flatten())	# calculate average with the same unique index 
# 	# print(similarity_average)
# 	rand = np.random.random(similarity_average.size)
# 	idx_sorted = np.lexsort((rand,similarity_average))[::-1] #sort random index by similarity_average
# 	print(idx_sorted)
# 	similaritys = similarity_average[idx_sorted]
# 	codes = codes[idx_sorted][similaritys>0.8]

# 	print("-----idx_sorted: ", idx_sorted)
# 	print(similaritys)
# 	print(codes)
# 	print("001099008838" in codes)


@app.post("/api/deleteFailFacev2")
async def deleteFailFacev2():
	try:
		redisClient.delete("FaceFeatureFail2")
		redisClient.delete("FaceListCodeFail2")
		if os.path.exists(PATH_IMG_FAIL):
			shutil.rmtree(PATH_IMG_FAIL)
			os.mkdir(PATH_IMG_FAIL)
		return {"success": True}
	except Exception as e:
		tb_str = traceback.format_exc()
		LOGGER_APP.error(f"Traceback: {tb_str}")
		return {"success": False, "error_code": 8008, "error": str(e)}

if __name__=="__main__":
	host = "0.0.0.0"
	port = 8423

	uvicorn.run("controller:app", host=host, port=port, log_level="info", reload=True)


"""
8000: "Don't have any registered user"
8001: "Don't find any face"
8002: "Fake face image"
8003: "Don't find any user"
8004: "This user has been registered!"
8005: "No users have been registered!"
8006: "This user has not been registered!"
8007: "Too many faces in this image"
8008: error system
8009: "Face size is not true"
8010: "There are some special letters in user code!"
"""


# docker run -it --shm-size=4g --rm -p8000:8000 -p8001:8001 -p8002:8002 -e PYTHONIOENCODING=UTF-8 -v ${PWD}:/workspace/ -v ${PWD}/my_repository:/models -v ${PWD}/requirements.txt:/opt/tritonserver/requirements.tx tritonserver_mq

# tritonserver --model-repository=/models --model-control-mode=poll --repository-poll-secs=5
