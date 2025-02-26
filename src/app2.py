import redis
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Request, Depends, Body
from fastapi.responses import StreamingResponse, JSONResponse

from schemes import *
from libs.utils import *

from service_ai.spoof_detection_onnx import FakeFace
from service_ai.retinanet_det import RetinanetRunnable
from service_ai.ghostface_onnx import GhostFaceRunnable

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
IMG_AVATAR = "static/avatar"
PATH_IMG_AVATAR = f"{str(ROOT)}/{IMG_AVATAR}"
PATH_IMG_FAIL = f"{str(ROOT)}/static/failface"
PATH_IMG_SPOOFING = f"{str(ROOT)}/static/spoofing"
PATH_LOG = f"{str(ROOT)}/logs"
PATH_IMG_REC = f"{str(ROOT)}/static/recognition"
check_folder_exist(path_avatar=PATH_IMG_AVATAR, path_imgfail=PATH_IMG_FAIL, path_log=PATH_LOG, path_rec=PATH_IMG_REC)
LOGGER_APP = set_log_file(file_name="app")

CONFIG_FACEDET = {
	"model_path": f"{str(ROOT)}/weights/detectFace_model_op16.onnx",
	"min_sizes": [[16,32], [64,128], [256,512]],
	"steps": [8, 16, 32],
	"variance": [0.1, 0.2],
	"clip": False,
	"conf_thres": 0.75,
	"iou_thres": 0.25,
	"image_size": [640,640],
	"device": "cpu",
}

CONFIG_GHOSTFACE = {
	"model_path": f"{str(ROOT)}/weights/ghostface.onnx",
	"imgsz": [112,112],
	"conf_thres": 0.75,
	"device": 'cpu',
}

# FACEDET = RetinanetRunnable(**CONFIG_FACEDET)
# GHOSTFACE = GhostFaceRunnable(**CONFIG_GHOSTFACE)
# SPOOFINGDET = FakeFace(f"{str(ROOT)}/weights/spoofing.onnx")

REDISSERVER_IP = os.getenv('REDISSERVER_IP', "192.168.6.163")
REDISSERVER_PORT = os.getenv('REDISSERVER_PORT', 6400)
logger.info(f"----REDISSERVER_IP: {REDISSERVER_IP}")
logger.info(f"----REDISSERVER_PORT: {REDISSERVER_PORT}")
redisClient = redis.StrictRedis(host=REDISSERVER_IP,
								port=int(REDISSERVER_PORT),
								password="RedisAuth",
								db=0)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    MULTIW.worker_pool = create_worker_pool(settings.max_workers)
    MULTIW.current_workers = settings.max_workers
    
    # Start background tasks
    task_processor = asyncio.create_task(process_tasks())
    auto_scaler = asyncio.create_task(auto_scale_workers())
    
    logger.info(f"Service started with {settings.max_workers} workers")
    
    yield
    
    # Shutdown
    logger.info("Shutting down service...")
    task_processor.cancel()
    auto_scaler.cancel()
    
    # Clean up the worker pool
    if MULTIW.worker_pool:
        MULTIW.worker_pool.shutdown(wait=True)
    
    logger.info("Service shutdown complete")

# Create FastAPI application
app = FastAPI(
    title="Face Recognition API",
    description="High-concurrency API for face recognition in images",
    version="1.0.0",
    lifespan=lifespan
)
app.mount("/static", StaticFiles(directory="static"), name="static")
# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)