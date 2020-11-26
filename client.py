#!/usr/bin/env python
import os
from clients.sbt_suit_client import sbt_suit_client
from clients.yolov5client import yolov5client
import cv2

# 获取client
triton_client = yolov5client()
sbt_client = sbt_suit_client()
for name in os.listdir("input"):
    # 推断
    result = triton_client .infer("input/"+name,triton_client,0.5)
    for obj in result:
        
        if (obj[0]==0):
            # print (obj)
            print(sbt_client.infer("input/"+name,obj,0.5),"shuangbaotai ")
    # print(result)



   
