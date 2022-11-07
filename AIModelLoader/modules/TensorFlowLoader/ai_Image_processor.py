import logging
from PIL import Image, ImageDraw
import io
import numpy as np
# import cv2 as cv
import json
import tensorflow as tf
import cv2
import numpy as np
import pickle
# from tensorflow.compat.v1 import ConfigProto
from tensorflow.python.saved_model import tag_constants
from ai_model_path import AI_Model_Path

class ProcessImages():
    
    def detect_object(self, imgBytes):

        results = []
        try:
            model_full_path = AI_Model_Path.Get_Model_Path()
            if(model_full_path == ""):
                logging.info("################ PLEASE SET AI MODEL FIRST")
                logging.info("############## ############## END ############## ##############")
                raise Exception ("PLEASE SET AI MODEL FIRST")
            logging.info("############## AI MODEL PATH: " + model_full_path)
            if '.tflite' in model_full_path:
                interpreter = tf.lite.Interpreter(model_path=model_full_path)
                interpreter.allocate_tensors()
                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()
                print("############## input_details ##############")
                print(input_details)
                print("############## output_details ##############")
                print(output_details)
                print()
                input_shape = input_details[0]['shape']

                # bytes to numpy.ndarray
                im_arr = np.frombuffer(imgBytes, dtype=np.uint8)  # im_arr is one-dim Numpy array
                img = cv2.imdecode(im_arr, flags=cv2.IMREAD_COLOR)

                im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                im_rgb = cv2.resize(im_rgb, (input_shape[1], input_shape[2]))
                input_data = np.expand_dims(im_rgb, axis=0)
                # input_data = np.asarray(input_data).astype(np.float32)
                print("############## input_data shape ##############")
                print(input_data.shape)
                print()

                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()
                output_data = interpreter.get_tensor(output_details[0]['index'])
                print("############## output_data shape ##############")
                print(output_data.shape)
                print()
                detection_boxes = interpreter.get_tensor(output_details[0]['index'])
                detection_classes = interpreter.get_tensor(output_details[1]['index'])
                detection_scores = interpreter.get_tensor(output_details[2]['index'])
                num_boxes = interpreter.get_tensor(output_details[3]['index'])

                label_names = [line.rstrip('\n') for line in open(AI_Model_Path.Get_Labelmap_Path())]
                label_names = np.array(label_names)
                new_label_names = list(filter(lambda x : x != '???', label_names))

                for i in range(int(num_boxes[0])):
                    if detection_scores[0, i] > .5:
                        class_id = int(detection_classes[0, i])
                        class_name = new_label_names[class_id]
                        # top,	left,	bottom,	right
                        results_json = "{'Class': '%s','Score': '%s','Location': '%s'}" % (class_name, detection_scores[0, i],detection_boxes[0, i])
                        results.append(results_json)
                        print(results_json)
        except Exception as e:
            print ( "detect_object unexpected error %s " % e )
            raise

        # return results
        return json.dumps(results)
    

class AIImageProcessor():
    def __init__(self):
        return

    def process_images(self, imgBytes):      
        logging.info("############## ############## Start Load AI Model ############## ##############")
        logging.info("############## tensorflow version is: " + tf.__version__ )

        resp = ProcessImages().detect_object(imgBytes)
             
        logging.info("############## detect result: " + resp)
        logging.info("############## ############## END ############## ##############")

        return resp
