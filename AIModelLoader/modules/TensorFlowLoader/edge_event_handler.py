import requests
import zipfile
import os
import hashlib, base64
from main import client
from ai_model_path import AI_Model_Path
import json
import logging

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
        # print ( "FilePath is: %s\n" % FilePath )
        # print ( "DownloadUrl is: %s\n" % DownloadUrl )
        FilePath = "/iotedge/storage/" + FileName
        # download AI model
        r = requests.get(DownloadUrl)
        print ("######## download AI Model Succeeded.")
        ffw = open(FilePath, 'wb')
        ffw.write(r.content)
        ffw.close()
        print ("######## AI Model File: " + FilePath)

        # # message to iot hub
        # print ( "######## Send message to iot hub")
        # await module_client.send_message_to_output("message from AML module FilePath is: " + FilePath, "IoTHubMeg")

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
            
            local_model_path = targeDir + "/ssd_mobilenet_v1_1_metadata_1.tflite"
            local_labelmap_path = targeDir + "/labelmap.txt"

            AI_Model_Path.Set_Model_Path(local_model_path)
            AI_Model_Path.Set_Labelmap_Path(local_labelmap_path)

            logging.info("############## SET AI MODEL PATH: " + local_model_path)
            logging.info("############## SET AI Labelmap PATH: " + local_labelmap_path)

            # message to module
            if client is not None:
                print ( "######## Send AI Model Info AS Routing Message")
                data = "{\"local_model_path\": \"%s\",\"local_labelmap_path\": \"%s\"}" % (filenamenoext+"/ssd_mobilenet_v1_1_metadata_1.tflite", filenamenoext+"/labelmap.txt")
                await client.send_message_to_output(data, "DLModelOutput")
                # update the reported properties
                reported_properties = {"LatestAIModelFileName": FileName }
                print("######## Setting reported LatestAIModelName to {}".format(reported_properties["LatestAIModelFileName"]))
                await client.patch_twin_reported_properties(reported_properties)
        else:
            print ( "######## New AI Model MD5 checksum failed")

    except Exception as ex:
        print ( "Unexpected error in twin_patch_handler: %s" % ex )

    # Define function for handling received messages

async def receive_message_handler(message):
    print("Message received")
    size = len(message.data)
    message_text = message.data.decode('utf-8')
    print("    Data: <<<{data}>>> & Size={size}".format(data=message.data, size=size))
    print("    Properties: {}".format(message.custom_properties))

    if message.input_name == "AIMessageInput":
        message_json = json.loads(message_text)
        
        local_model_path = "/var/lib/videoanalyzer/" + message_json["local_model_path"]
        local_labelmap_path = "/var/lib/videoanalyzer/" + message_json["local_labelmap_path"]

        AI_Model_Path.Set_Model_Path(local_model_path)
        AI_Model_Path.Set_Labelmap_Path(local_labelmap_path)

        logging.info("############## SET AI MODEL PATH: " + local_model_path)
        logging.info("############## SET AI Labelmap PATH: " + local_labelmap_path)

def get_filename_and_ext(filename):
    (filepath,tempfilename) = os.path.split(filename);
    (shotname,extension) = os.path.splitext(tempfilename);
    return shotname,extension

def unzip_file(unZipSrc,targeDir):
    if not os.path.isfile(unZipSrc):
        raise Exception(u'unZipSrc not exists:{0}'.format(unZipSrc))
    if not os.path.isdir(targeDir):
        os.makedirs(targeDir)
    print(u'######## Start decompressing the file: {0}'.format(unZipSrc))
    unZf = zipfile.ZipFile(unZipSrc,'r')
    for name in unZf.namelist() :
        unZfTarge = os.path.join(targeDir,name)

        if unZfTarge.endswith("/"):
            #empty dir
            splitDir = unZfTarge[:-1]
            if not os.path.exists(splitDir):
                os.makedirs(splitDir)
        else:
            splitDir,_ = os.path.split(targeDir)
            if not os.path.exists(splitDir):
                os.makedirs(splitDir)
            hFile = open(unZfTarge,'wb')
            hFile.write(unZf.read(name))
            hFile.close()
    print(u'######## Unzipped. Target file directory: {0}'.format(targeDir))
    unZf.close()

def content_encoding(path: str):
    with open(path, 'rb') as f:
        content = f.read()
    content_md5 = hashlib.md5()
    content_md5.update(content)
    content_base64 = base64.b64encode(content_md5.digest())
    return content_base64.decode("utf-8")