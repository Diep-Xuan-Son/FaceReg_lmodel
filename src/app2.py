import redis
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Request, Depends, Body, HTTPException, status
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.encoders import jsonable_encoder
from contextlib import asynccontextmanager

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

