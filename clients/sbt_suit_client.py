
import argparse
import numpy as np
import sys
import cv2

import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException

from processing import preprocess, postprocess,preprocess_suit_img
from render import render_box, render_filled_box, get_text_size, render_text, RAND_COLORS
from labels import COCOLabels

import time
import multiprocessing
from multiprocessing import Pool
import os
class sbt_suit_client():
    url = '192.168.1.221:8001'
    model = 'resnet18-sbt'
    ssl = False
    root_certificates = None
    private_key = None
    certificate_chain = None
    client_timeout = None
    nms = 0.4
    model_info = False
    verbose = False
    
    def __init__(self):
        # Create server context
        try:
            triton_client = grpcclient.InferenceServerClient(
                url=self.url,
                verbose=self.verbose,
                ssl=self.ssl,
                root_certificates=self.root_certificates,
                private_key=self.private_key,
                certificate_chain=self.certificate_chain)
        except Exception as e:
            print("context creation failed: " + str(e))
            sys.exit()

        # Health check
        if not triton_client.is_server_live():
            print("FAILED : is_server_live")
            sys.exit(1)

        if not triton_client.is_server_ready():
            print("FAILED : is_server_ready")
            sys.exit(1)

        if not triton_client.is_model_ready(self.model):
            print("FAILED : is_model_ready")
            sys.exit(1)


        try:
            metadata = triton_client.get_model_metadata(self.model)
            # print(metadata)
        except InferenceServerException as ex:
            if "Request for unknown model" not in ex.message():
                print("FAILED : get_model_metadata")
                print("Got: {}".format(ex.message()))
                sys.exit(1)
            else:
                print("FAILED : get_model_metadata")
                sys.exit(1)

        # Model configuration
        try:
            config = triton_client.get_model_config(self.model)
            if not (config.config.name ==self.model):
                print("FAILED: get_model_config")
                sys.exit(1)
            # print(config)
        except InferenceServerException as ex:
            print("FAILED : get_model_config")
            print("Got: {}".format(ex.message()))
            sys.exit(1)
        self.triton_client = triton_client
        

    def infer(self,input_img,obj,confidence):
        confidence = confidence
        out = "crop/"+input_img.split("/")[1]
        # IMAGE MODE
        # print("Running in 'image' mode")
        if not input_img:
            print("FAILED: no input image")
            sys.exit(1)
        
        inputs = []
        outputs = []
        inputs.append(grpcclient.InferInput('data', [1, 3,128,64], "FP32"))
        outputs.append(grpcclient.InferRequestedOutput('prob'))

        # print("Creating buffer from image file...")
        input_image = cv2.imread(input_img)
        h= int(input_image.shape[0])
        w = int(input_image.shape[1])
        input_image = input_image[int(h*obj[3]):int(h*obj[4]), int(w*obj[1]):int(w*obj[2])]
        print(obj)
        cv2.imwrite(out, input_image)
        if input_image is None:
            print(f"FAILED: could not load input image {str(input_img)}")
            sys.exit(1)
        input_image_buffer = preprocess_suit_img(input_image,128,64)
        input_image_buffer = np.expand_dims(input_image_buffer, axis=0)
        inputs[0].set_data_from_numpy(input_image_buffer)

        # print("Invoking inference...")
        results = self.triton_client.infer(model_name=self.model,
                                    inputs=inputs,
                                    outputs=outputs,
                                    client_timeout=self.client_timeout)
        result = results.as_numpy('prob')
        # print (result[0])
        return np.argmax(result[0])
