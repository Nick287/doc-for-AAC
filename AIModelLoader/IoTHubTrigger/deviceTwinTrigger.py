# from msrest.authentication import ApiKeyCredentials
# import requests
import time
# import sys
from time import sleep
from azure.iot.hub import IoTHubRegistryManager
from azure.iot.hub.models import Twin, TwinProperties, QuerySpecification, QueryResult
from azure.iot.hub.protocol.operations.devices_operations import DevicesOperations


import requests

# bo AVA
iothub_connection_str = "HostName=avasample76havxesh5rb6.azure-devices.net;SharedAccessKeyName=iothubowner;SharedAccessKey=q/8JCYncr2r1JS3CQCJVat5FpLl/2zA3SIoWZ6XShT0="
device_id = "avasample-iot-edge-device"
# module_id = "APIModel"

module_id = "DownloaderModule"


# iothub_connection_str = "HostName=avasample76havxesh5rb6.azure-devices.net;SharedAccessKeyName=iothubowner;SharedAccessKey=q/8JCYncr2r1JS3CQCJVat5FpLl/2zA3SIoWZ6XShT0="
# device_id = "EdgeVM001"
# # module_id = "TensorFlowLoader"

# module_id = "ONNXLoader"

# RegistryManager
iothub_registry_manager = IoTHubRegistryManager.from_connection_string(iothub_connection_str)
module_twin = iothub_registry_manager.get_module_twin(device_id,module_id)


# module_twin.properties.desired["FileName"]      = "tensorflow001.zip"
# module_twin.properties.desired["DownloadUrl"]   = "https://avasample76havxesh5rb6.blob.core.windows.net/testaidownload/tensorflow001.zip?sp=r&st=2021-11-25T14:32:49Z&se=2022-09-28T22:32:49Z&sv=2020-08-04&sr=b&sig=INTK5SNsvKU%2FYTlYkg7l11pTGjRFWvWk2T1MPbKS0QQ%3D"
# module_twin.properties.desired["ContentMD5"]    = "wPJ4M5M4BVAkuk6fdJYODg=="


# module_twin.properties.desired["FileName"]      ="tiny-yolov3-11.zip"
# module_twin.properties.desired["DownloadUrl"]   ="https://avasample76havxesh5rb6.blob.core.windows.net/testaidownload/tiny-yolov3-11.zip?sp=r&st=2021-12-12T13:33:30Z&se=2022-10-11T21:33:30Z&sv=2020-08-04&sr=b&sig=3R4MgDtQtBX6rhf0J0pvwYiGduxnJfeAr%2BRWGaK3%2Bs4%3D"
# module_twin.properties.desired["ContentMD5"]    ="/htntB5xaQkqrrqvA6ps6A=="

module_twin.properties.desired["FileName"]      = "tiny-yolov4-416-chatswood-entrance-camera_best_3.zip"
module_twin.properties.desired["DownloadUrl"]   = "https://svaesvmodstgacctssdev.blob.core.windows.net/amlmodels/tiny-yolov4-416-chatswood-entrance-camera_best_3.zip?sv=2020-10-02&st=2021-12-14T02%3A32%3A47Z&se=2021-12-14T02%3A42%3A47Z&sr=b&sp=r&sig=q%2FLlAJmcsDhXEliCNbeom2s1ShcMkTcVNOnHE1qX%2Bcw%3D&md5=N9Rv98+kI8VdZ5efNQqyDw=="
module_twin.properties.desired["ContentMD5"]    = "N9Rv98+kI8VdZ5efNQqyDw=="

# DownloadUrl = "https://avasample76havxesh5rb6.blob.core.windows.net/testaidownload/tiny-yolov3-11.zip?sp=r&st=2021-12-12T13:33:30Z&se=2022-10-11T21:33:30Z&sv=2020-08-04&sr=b&sig=3R4MgDtQtBX6rhf0J0pvwYiGduxnJfeAr%2BRWGaK3%2Bs4%3D"

# DownloadUrl = "https://svaesvmodstgacctssdev.blob.core.windows.net/amlmodels/tiny-yolov4-416-chatswood-entrance-camera_best_3.zip?sv=2020-10-02&st=2021-12-14T02%3A32%3A47Z&se=2021-12-14T02%3A42%3A47Z&sr=b&sp=r&sig=q%2FLlAJmcsDhXEliCNbeom2s1ShcMkTcVNOnHE1qX%2Bcw%3D&md5=N9Rv98+kI8VdZ5efNQqyDw=="
# FilePath = "D:\CSE Project\Share\az-iotedge-ai-model-loader\AIModelLoader\IoTHubTrigger" + "\tiny-yolov4-416-chatswood-entrance-camera_best_3.zip"

# r = requests.get(DownloadUrl)
# print ("######## response code: " + str(r.status_code))
# print ("######## download AI Model Succeeded.")
# ffw = open(FilePath, 'wb')
# ffw.write(r.content)
# ffw.close()
# print ("######## AI Model File: " + FilePath)


iothub_registry_manager.update_module_twin(device_id,module_id,module_twin,module_twin.etag)

time.sleep(2)
module_twin = iothub_registry_manager.get_module_twin(device_id,module_id)
reportedTwins = module_twin.properties.reported

# if module_twin.properties.reported["LatestAIModelName"] is not None:
#     print(module_twin.properties.reported["LatestAIModelName"])

if "LatestAIModelName" in reportedTwins:
    print(reportedTwins["LatestAIModelName"])