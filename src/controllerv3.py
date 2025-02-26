import torch
import asyncio
import uvicorn
# import redis
import logging
import threading
import traceback
from io import BytesIO
import multiprocessing
from string import ascii_letters, digits, punctuation

from app2 import *
from starlette.status import HTTP_202_ACCEPTED
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor

# Configure logging
LOGGER = logging.getLogger()
LOGGER.setLevel(logging.DEBUG)
FILE_HANDLER = logging.FileHandler("./logs/myapp.log", "w")
FILE_HANDLER.setLevel(logging.DEBUG)
LOG_FORMAT = 'AICLOUD: %(name)s - %(levelname)s - %(message)s'
FILE_LOG_FORMAT = '%(asctime)s.%(msecs)03d - ' + LOG_FORMAT
FILE_HANDLER.setFormatter(logging.Formatter(
	FILE_LOG_FORMAT,
	datefmt='%Y-%m-%d %H:%M:%S'
))
LOGGER.addHandler(FILE_HANDLER)

# Configuration settings
class Config:
    # GPU settings
    CUDA_AVAILABLE = torch.cuda.is_available()
    NUM_GPUS = torch.cuda.device_count() if CUDA_AVAILABLE else 0
    
    # Worker pool settings
    MIN_WORKERS = 1
    MAX_WORKERS = 5  # Maximum number of workers to scale to
    INITIAL_WORKERS = 2
    
    # Auto-scaling settings
    SCALING_THRESHOLD_HIGH = 0.8  # Scale up when queue is 80% full
    SCALING_THRESHOLD_LOW = 0.2   # Scale down when queue is 20% full
    SCALING_CHECK_INTERVAL = 10   # Check for scaling every 10 seconds
    SCALE_COOLDOWN = 30
    
    # Performance settings
    REQUEST_TIMEOUT = 30  # Seconds
    QUEUE_MAX_SIZE = 5000
    
    # Person detection settings
    CONFIDENCE_THRESHOLD = 0.5

    BATCH_SIZE = 8


class MultiWoker():
	def __init__(self, ):
		# Task queue and results
		self.request_queue = asyncio.Queue()
		self.results_store: Dict[str, Dict] = {}

		# Worker pool
		self.worker_pool = None
		self.last_scale_time = 0
		self.current_workers = Config.MAX_WORKERS

		self.worker_lock = threading.Lock()
		self.current_worker_count = Config.INITIAL_WORKERS
		self.workers = []

	@classmethod
	# Create process pool for workers
	def create_worker_pool(num_workers):
	    return ThreadPoolExecutor(max_workers=num_workers)

	def add_worker(self, new_workers, current_time):
        with self.worker_lock:
            if self.current_worker_count >= Config.MAX_WORKERS:
                logger.info("Max workers reached, not adding more")
                return False
                
            # Determine which GPU to use (round-robin)
            device_id = None
            if Config.CUDA_AVAILABLE and Config.NUM_GPUS > 0:
                device_id = self.current_worker_count % Config.NUM_GPUS
            
            try:
                # Create new detector
                detector = PersonDetector(device_id)
                
                # Add to worker list
                self.workers.append(detector)
                self.current_worker_count += 1

                # Create new worker pool
                self.worker_pool = self.create_worker_pool(new_workers)
                self.current_workers = new_workers
                self.last_scale_time = current_time
                
                logger.info(f"Added worker #{self.current_worker_count} on device {device_id if device_id is not None else 'CPU'}")
                return True
            except Exception as e:
            	tb_str = traceback.format_exc()
                logger.error(f"Failed to add worker: {tb_str}")
                return False

    def remove_worker(self, new_workers, current_time):
        with worker_lock:
            if self.current_worker_count <= Config.MIN_WORKERS:
                logger.info("Min workers reached, not removing more")
                return False
            
            # Remove a worker (just decrement the count, the actual worker will be garbage collected)
            if self.workers:
                self.workers.pop()
                self.current_worker_count -= 1

                # Create new worker pool
                self.worker_pool = self.create_worker_pool(new_workers)
                self.current_workers = new_workers
                self.last_scale_time = current_time
                logger.info(f"Removed worker, now at {self.current_worker_count}")
                return True
            return False

	# Worker utilization metric
	def get_worker_utilization(self,):
	    if self.worker_pool is None:
	        return 0.0
	    active_workers = len([1 for t in self.worker_pool._threads if t.is_alive()])
	    return active_workers / self.current_workers

	# Auto-scaling logic
	async def auto_scale_workers(self,):
	    while True:
	        try:
	            utilization = self.get_worker_utilization()
	            queue_size = self.request_queue.qsize()
	            current_time = time.time()
	            
	            logger.info(f"Worker utilization: {utilization:.2f}, Queue size: {queue_size}")
	            
	            # Scale up if high utilization and not recently scaled
	            if (utilization > Config.SCALING_THRESHOLD_HIGH or queue_size > self.current_workers * 3) and \
	               self.current_workers < Config.MAX_WORKERS and \
	               (current_time - self.last_scale_time) > Config.SCALE_COOLDOWN:
	                
	                new_workers = min(self.current_workers * 2, Config.MAX_WORKERS)
	                logger.info(f"Scaling up workers from {self.current_workers} to {new_workers}")
	                
	                # Replace the worker pool
	                old_pool = self.worker_pool
	                self.add_worker(new_workers, current_time)
	                
	                # Shutdown old pool gracefully
	                if old_pool:
	                    old_pool.shutdown(wait=False)
	            
	            # Scale down if low utilization and not recently scaled
	            elif utilization < Config.SCALING_THRESHOLD_LOW and queue_size < self.current_workers and self.current_workers > Config.MAX_WORKERS and \
	                 (current_time - self.last_scale_time) > Config.SCALE_COOLDOWN:
	                
	                new_workers = max(self.current_workers // 2, Config.MAX_WORKERS)
	                logger.info(f"Scaling down workers from {self.current_workers} to {new_workers}")
	                
	                # Replace the worker pool
	                old_pool = self.worker_pool
	                self.remove_worker(new_workers, current_time)
	                
	                # Shutdown old pool gracefully
	                if old_pool:
	                    old_pool.shutdown(wait=False)
	                
	        except Exception as e:
	            logger.error(f"Error in auto-scaling: {str(e)}")
	        
	        await asyncio.sleep(Config.SCALING_CHECK_INTERVAL)  # Check every 5 seconds

	# Process image batch
	def process_batch(model, image_batch, task_ids):
	    try:
	        # Convert image batch to tensor format expected by the model
	        tensor_batch = [torch.from_numpy(img).permute(2, 0, 1).float().div(255.0) for img in image_batch]
	        tensor_batch = torch.stack(tensor_batch)
	        
	        # Run inference
	        with torch.no_grad():
	            tensor_batch = tensor_batch.to(f'cuda:{Config.NUM_GPUS}' if torch.cuda.is_available() else 'cpu')
	            predictions = model(tensor_batch)
	        
	        # Process results
	        results = predictions.pandas().xyxy
	        
	        # Store results for each task
	        for i, task_id in enumerate(task_ids):
	            # Filter for person class (class 0 in COCO dataset)
	            person_detections = results[i][results[i]['class'] == 0]
	            person_detections = person_detections[person_detections['confidence'] >= Config.CONFIDENCE_THRESHOLD]
	            
	            results_store[task_id] = {
	                'status': 'completed',
	                'detections': person_detections[['xmin', 'ymin', 'xmax', 'ymax', 'confidence']].to_dict('records'),
	                'person_count': len(person_detections),
	                'completed_at': time.time()
	            }
	            
	            logger.info(f"Task {task_id} completed with {len(person_detections)} person detections")
	    
	    except Exception as e:
	        for task_id in task_ids:
	            results_store[task_id] = {
	                'status': 'error',
	                'error': str(e),
	                'completed_at': time.time()
	            }
	        tb_str = traceback.format_exc()
	        logger.error(f"Batch processing error: {tb_str}")

	# Process image batch
	def process_batch(self, model, image_batch, task_ids):

	# Task processor
	async def process_tasks():
	    # Load model
	    # model = load_model()
	    # logger.info(f"Model loaded on {'GPU' if torch.cuda.is_available() else 'CPU'}")
	    facedet = RetinanetRunnable(**CONFIG_FACEDET)
		ghostface = GhostFaceRunnable(**CONFIG_GHOSTFACE)
		spoofingdet = FakeFace(f"{str(ROOT)}/weights/spoofing.onnx")
	    
	    # Process batches from the queue
	    while True:
	        try:
	            batch_tasks = []
	            batch_images = []
	            batch_task_ids = []
	            
	            # Get the first task
	            task = await self.request_queue.get()
	            batch_tasks.append(task)
	            batch_images.append(task['image'])
	            batch_task_ids.append(task['task_id'])
	            
	            # Try to get more tasks to fill the batch
	            try:
	                for _ in range(Config.BATCH_SIZE - 1):
	                    if self.request_queue.qsize() > 0:
	                        task = self.request_queue.get_nowait()
	                        batch_tasks.append(task)
	                        batch_images.append(task['image'])
	                        batch_task_ids.append(task['task_id'])
	                    else:
	                        break
	            except asyncio.QueueEmpty:
	                pass
	            
	            logger.info(f"Processing batch of {len(batch_tasks)} tasks")
	            
	            # Process batch in thread pool
	            if self.worker_pool is not None:
	                self.worker_pool.submit(process_batch, model, batch_images, batch_task_ids)
	            
	            # Mark tasks as done in the queue
	            for _ in batch_tasks:
	                self.request_queue.task_done()
	                
	        except Exception as e:
	        	tb_str = traceback.format_exc()
	            logger.error(f"Error in task processor: {tb_str}")
	            await asyncio.sleep(1)  # Prevent tight loop in case of errors

MULTIW = MultiWoker()

# Response models
class SearchUserResponse(BaseModel):
    task_id: str
    status: str
    message: str

class SearchUserResult(BaseModel):
    status: str
    detections: Optional[List[DetectionResult]] = None
    person_count: Optional[int] = None
    error: Optional[str] = None
    completed_at: float



@app.post("/api/searchUser", status_code=HTTP_202_ACCEPTED, response_model=SearchUserResponse)
async def searchUser(image: UploadFile = File(...)):
	try:
		# Generate task ID
        task_id = str(uuid.uuid4())

        # Read and preprocess image
        image_data = await file.read()
        nparr = np.frombuffer(image_data, np.uint8)
    	img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        # Create task and add to queue
        MULTIW.results_store[task_id] = {'status': 'pending'}
        await MULTIW.request_queue.put({'task_id': task_id, 'image': img, 'submitted_at': time.time()})

        queue_size = MULTIW.request_queue.qsize()
        logger.info(f"Task {task_id} queued. Current queue size: {queue_size}")

        return SearchUserResponse(
            task_id=task_id,
            status="pending",
            message=f"Task submitted and queued. Current queue size: {queue_size}"
        )

	except Exception as e:
		tb_str = traceback.format_exc()
        logger.error(f"Error submitting task: {tb_str}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.get("/api/getSearchUserResult", response_model=SearchUserResult)
async def getSearchUserResult(task_id: str):
    """
    Retrieve the results of a detection task by task ID.
    """
    if task_id not in MULTIW.results_store:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return MULTIW.results_store[task_id]

@app.get("/health")
async def get_health_status():
    """
    Get service health status and metrics.
    """
    return {
        "status": "healthy",
        "worker_count": MULTIW.current_workers,
        "worker_utilization": self.get_worker_utilization(),
        "queue_size": MULTIW.request_queue.qsize(),
        "active_tasks": len(MULTIW.results_store),
        "gpu_available": torch.cuda.is_available(),
    }

#---------------------old-----------------------------
heart_beat_thread = threading.Thread(target=delete_file_cronj, args=(PATH_IMG_FAIL, 25200), daemon=True)https://github.com/Diep-Xuan-Son/FaceReg_lmodel
heart_beat_thread.start()

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
