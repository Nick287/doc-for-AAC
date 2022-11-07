from datetime import datetime
import io
import json
import os
import logging
import time
from flask import Flask, Response, Request, abort, request
import requests
import argparse
from ai_Image_processor import AIImageProcessor

import sys
import asyncio
import signal
import threading
from azure.iot.device.aio import IoTHubModuleClient
from ai_model_path import AI_Model_Path
import edge_event_handler 

client = None

def IoT_Edge_Start():
    if not sys.version >= "3.5.3":
        raise Exception( "The sample requires python 3.5.3+. Current version of Python: %s" % sys.version )
    print ( "IoT Hub Client for Python" )

    # NOTE: Client is implicitly connected due to the handler being set on it
    client = create_client()

    # Event indicating client stop
    stop_event = threading.Event()

    # Define a handler to cleanup when module is is terminated by Edge
    def module_termination_handler(signal, frame):
        print ("IoTHubClient sample stopped by Edge")
        stop_event.set()

    # Set the Edge termination handler
    signal.signal(signal.SIGTERM, module_termination_handler)

    # Run the sample
    loop = asyncio.get_event_loop()
    try:
        # loop.run_until_complete(run_sample(client))
        loop.run_until_complete(asyncio.gather(run_sample(client), Run_WebAPI()))
    except Exception as e:
        print("Unexpected error %s " % e)
        raise
    finally:
        print("Shutting down IoT Hub Client...")
        loop.run_until_complete(client.shutdown())
        loop.close()

async def run_sample(client):
    # Customize this coroutine to do whatever tasks the module initiates
    # e.g. sending messages
    while True:
        await asyncio.sleep(1000)

def create_client():
    client = IoTHubModuleClient.create_from_edge_environment()

    try:
        # Set handler on the client
        client.on_message_received = edge_event_handler.receive_message_handler
        # set the twin patch handler on the client
        client.on_twin_desired_properties_patch_received = edge_event_handler.twin_patch_handler
        logging.info('message_handler listening ...')
    except:
        # Cleanup if failure occurs
        client.shutdown()
        raise

    return client

def Run_WebAPI():
    # Get application arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('-p', nargs=1, metavar=('http_server_port'), help='Port number to listen on.', type=int, default=8080)

    _arguments = parser.parse_args()

    # Default to port 8080
    httpServerPort = _arguments.p

    app = Flask(__name__)

    # init_logging(app)

    processor = AIImageProcessor()
    logging.info('Http extension listening on port: {}'.format(httpServerPort))

    # /score routes to scoring function 
    # This function returns a JSON object with inference result
    @app.route("/score", methods=['POST'])
    def score():
        try:
            image_data = request.get_data()
            
            result = processor.process_images(image_data)

            if(result is not None):
                # respBody = {
                #     "inferences" : result
                # }
                # respBody = json.dumps(respBody)
                return Response(result, status = 200, mimetype ='application/json')
            else:
                return Response(status = 400)
        except Exception as ex:
            logging.info('error: {}'.format(ex))
            abort(Response(response=str(ex), status = 400))
    # Running the file directly
    app.run(host='0.0.0.0', port=httpServerPort)

if __name__ == '__main__':
    # Set logging parameters
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)-15s] [%(threadName)-12.12s] [%(levelname)s]: %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)       # write in stdout
        ]
    )
    IoT_Edge_Start()