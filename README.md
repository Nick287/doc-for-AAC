# How To Enable ML Inference Modules On Azure IoT Edge Device

## Background

One of the most popular Edge scenarios is the artificial intelligence (AI) on Edge (Image Classification, Object Detection, Body, Face & Gesture Analysis, Image Manipulation, ETC ... ). Azure IoT Edge can certainly support these scenarios, These AI ability can also be improve over the model update, but in some scenarios the Edge device network environment is not good, Especially for some manufacturing equipment such as wind power or oil exploitation which equipment in the desert or in the sea.

However, as you know, Azure IoT Edge Module is the basis on docker. In generally an Edge module image with the AI environment which size will be around GB level at least, so how to incremental updates the AI model in a narrow bandwidth network its becomes more meaningful. That's why this document was written. The idear is made a Edge AI loader module which can load *Object Detection* TensorFlow or ONNX AI models and enable this AI module as a WebAPI. So that in this way Edge module can benefits other application or module.

## Overview

This document can help you build a Azure IoT Edge Module can download AI model then enable Inference ability. support TensorFlow *.tflite format model or *.onnx (Open Neural Network Exchange) format model.

So for ML Inference Modules have some Key concepts need to be clarify first (If you are not familiar with AI, you can learn the basics knowledge of TensorFlow lite and ONNX in the following two sections).

### TensorFlow

1. AI Model File: *.tflite its pre-trained AI model which download from [TensorFlow.org - Download starter model with Metadata](https://www.tensorflow.org/lite/examples/object_detection/overview) and its a generic AI model format that can be used in cross-platform applications such as IOS and Android. And about more information about Metadata and associated fields (eg: labels.txt) see [Read the metadata from models](https://www.tensorflow.org/lite/convert/metadata#read_the_metadata_from_models)
2. Model description： An object detection model is trained to detect the presence and location of multiple classes of objects. For example, a model might be trained with images that contain various pieces of fruit, along with a label that specifies the class of fruit they represent (e.g. an apple, a banana, or a strawberry), and data specifying where each object appears in the image.

    When an image is subsequently provided to the model, it will output a list of the objects it detects, the location of a bounding box that contains each object, and a score that indicates the confidence that detection was correct.
3. In case of if you want to build or customize tuning an AI Model please see [TensorFlow Lite Model Maker](https://www.tensorflow.org/lite/guide/model_maker)
4. More free pre-trained detection models with a variety of latency and precision characteristics can be found in the [Detection Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md#mobile-models). Each one of them follows the input and output signatures described in the following sections.

### Open Neural Network Exchange (ONNX)

Open Neural Network Exchange (ONNX) is an open standard format for representing machine learning models. ONNX is supported by a community of partners who have implemented it in many frameworks and tools.

1. ONNX supports a variety of tools to Build and deploy models and Frameworks & Converters [Build / Deploy Model](https://onnx.ai/supported-tools.html)
2. You can use ONNX runtime run ONNX Pre-Trained Models [ONNX Model Zoo](https://github.com/onnx/models) for implementation Vision Language Other AI features
3. In this case you can use model of [Object Detection & Image Segmentation - Tiny YOLOv3](https://github.com/onnx/models/tree/master/vision/object_detection_segmentation/tiny-yolov3)


The dynamically loaded AI model base on the features of IoT Edge module Twin, the architecture please refer below and work steps:

1. Upload Pre-Trained AI Models to public blob storage (Or any other Web service, just for the Edge Module can access this resource and download to Edge device later)
2. The IoT Hub will sync device module twins automatically with AI Models information, the sync will be done even if IoT Edge offline for some time.
3. The Loader Module monitors the updates of module twins via SDK. Through this way, it can get the ML model SAS token, and then download the AI model.
4. The Loader Module saves the AI model in the shared local storage of the IoT Edge Module. The local storage needs to be configured in the IoT Edge deployment JSON file.
5. The Loader Module load the AI model from the local storage by TensorFlow/ONNX SDK.
6. The Loader Module starts a Web API that receives the binary photo via post request and returns results in json file

In case for update AI model we can upload new AI model to blob storage and sync device module twins again instead of update whole IoT Edge module image, so this is the AI model incremental updates.

![image](image/architecture_diagram.png)

### Download trained AI model

Here I recommend using device twin to receive notifications that a new model is ready. Even when the device is offline, the message can still be cached in the IoT Hub to wait for the Edge device come back, and the message will be automatically synchronized.

Here is a code example of the python code used to register notifications for device twin then download AI model as a zip package. And perform further operations on the downloaded file.

1. The device twin notification was received, which included the file name(File name You can include version information such as 1.0, 1.1, or 2.0), file download address, and MD5 authentication token.
2. Download AI model as zip file to local storage.
3. MD5 checksum (optional) MD5 verification is prevent zip files being tampered during the network transmission.
4. Unzip it and save it locally.
5. Send notifications to IoT Hub or routing message to inform that the new AI model is ready.

```python
# define behavior for receiving a twin patch
async def twin_patch_handler(patch):
    try:
        print( "######## The data in the desired properties patch was: %s" % patch)
        if "FileName" in patch:
            FileName = patch["FileName"]
        if "DownloadUrl" in patch:
            DownloadUrl = patch["DownloadUrl"]
        if "ContentMD5" in patch:
            ContentMD5 = patch["ContentMD5"]
        FilePath = "/iotedge/storage/" + FileName

        # download AI model
        r = requests.get(DownloadUrl)
        print ("######## download AI Model Succeeded.")
        ffw = open(FilePath, 'wb')
        ffw.write(r.content)
        ffw.close()
        print ("######## AI Model File: " + FilePath)

        # MD5 checksum
        md5str = content_encoding(FilePath)
        if md5str == ContentMD5:
            print ( "######## New AI Model MD5 checksum succeeded")
            # decompressing the ZIP file
            unZipSrc = FilePath
            targeDir = "/iotedge/storage/"
            filenamenoext = get_filename_and_ext(unZipSrc)[0]
            targeDir = targeDir + filenamenoext
            unzip_file(unZipSrc,targeDir)
            
            # ONNX
            local_model_path = targeDir + "/tiny-yolov3-11.onnx"
            local_labelmap_path = targeDir + "/coco_classes.txt"

            # TensorFlow flite
            # local_model_path = targeDir + "/ssd_mobilenet_v1_1_metadata_1.tflite"
            # local_labelmap_path = targeDir + "/labelmap.txt"

            # message to module
            if client is not None:
                print ( "######## Send AI Model Info AS Routing Message")
                data = "{\"local_model_path\": \"%s\",\"local_labelmap_path\": \"%s\"}" % (filenamenoext+"/tiny-yolov3-11.onnx", filenamenoext+"/coco_classes.txt")
                await client.send_message_to_output(data, "DLModelOutput")
                # update the reported properties
                reported_properties = {"LatestAIModelFileName": FileName }
                print("######## Setting reported LatestAIModelName to {}".format(reported_properties["LatestAIModelFileName"]))
                await client.patch_twin_reported_properties(reported_properties)
        else:
            print ( "######## New AI Model MD5 checksum failed")

    except Exception as ex:
        print ( "Unexpected error in twin_patch_handler: %s" % ex )
```

### Input to model

Resized image (1x3x416x416) Original image size (1x2) which is [image.size[1], image.size[0]]

### Output of model

The model has 3 outputs. boxes: (1x'n_candidates'x4), the coordinates of all anchor boxes, scores: (1x80x'n_candidates'), the scores of all anchor boxes per class, indices: ('nbox'x3), selected indices from the boxes tensor. The selected index format is (batch_index, class_index, box_index). The class list is [here](https://github.com/qqwweee/keras-yolo3/blob/master/model_data/coco_classes.txt)

### Preprocessing steps

```python
import numpy as np
from PIL import Image

# this function is from yolo3.utils.letterbox_image
def letterbox_image(image, size):
    '''resize image with unchanged aspect ratio using padding'''
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    return new_image

def preprocess(img):
    model_image_size = (416, 416)
    boxed_image = letterbox_image(img, tuple(reversed(model_image_size)))
    image_data = np.array(boxed_image, dtype='float32')
    image_data /= 255.
    image_data = np.transpose(image_data, [2, 0, 1])
    image_data = np.expand_dims(image_data, 0)
    return image_data

image = Image.open(img_path)
# input
image_data = preprocess(image)
image_size = np.array([image.size[1], image.size[0]], dtype=np.float32).reshape(1, 2)
```

## Reference

- [Understand and use module twins in IoT Hub](https://docs.microsoft.com/en-us/azure/iot-hub/iot-hub-devguide-module-twins)
- [Learn how to deploy modules and establish routes in IoT Edge](https://docs.microsoft.com/en-us/azure/iot-edge/module-composition?view=iotedge-2020-11)
- [Building Azure IoT Edge Module with Message Routing 101](https://tsmatz.wordpress.com/2019/10/19/azure-iot-hub-iot-edge-module-container-tutorial-with-message-route/)
- [Give modules access to a device's local storage](https://docs.microsoft.com/en-us/azure/iot-edge/how-to-access-host-storage-from-module?view=iotedge-2020-11#link-module-storage-to-device-storage)
- [Add local storage to Azure IoT Edge modules using Docker Bind](https://sandervandevelde.wordpress.com/2021/01/07/add-local-storage-to-azure-iot-edge-modules-using-docker-bind/)
- [Understand IoT Edge automatic deployments for single devices or at scale](https://docs.microsoft.com/en-us/azure/iot-edge/module-deployment-monitoring?view=iotedge-2020-11)
- [Open Neural Network Exchange](https://github.com/onnx/)
- [ONNX Tutorials](https://github.com/onnx/tutorials)
- [Deploy ML model on IoT and edge devices](https://github.com/microsoft/onnxruntime/blob/gh-pages/docs/tutorials/iot-edge.md)
