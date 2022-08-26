#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# https://github.com/pupil-labs/pyndsi/tree/v1.0
import ndsi  # Main requirement
import numpy as np

SENSOR_TYPES = ["video", "gaze"]
SENSORS = {}  # Will store connected sensors

rows_pupil = 1088
cols_pupil = 1080


def on_network_event(network, event):
    # Handle gaze sensor attachment
    if event["subject"] == "attach" and event["sensor_type"] in SENSOR_TYPES:
        # Create new sensor, start data streaming,
        # and request current configuration
        sensor = network.sensor(event["sensor_uuid"])
        sensor.set_control_value("streaming", True)
        sensor.refresh_controls()

        # Save sensor s.t. we can fetch data from it in main()
        SENSORS[event["sensor_uuid"]] = sensor
        #print(f"Added sensor {sensor}...")

    # Handle gaze sensor detachment
    if event["subject"] == "detach" and event["sensor_uuid"] in SENSORS:
        # Known sensor has disconnected, remove from list
        SENSORS[event["sensor_uuid"]].unlink()
        del SENSORS[event["sensor_uuid"]]
        #print(f"Removed sensor {event['sensor_uuid']}...")
        
        
def init_network():
        return ndsi.Network(formats={ndsi.DataFormat.V4}, callbacks=(on_network_event,))
    

def fetch_sensor_data():
    world_img = np.zeros((rows_pupil, cols_pupil, 3))
    gaze = (0, 0)
    
    # Iterate over all connected devices
    for sensor in SENSORS.values():

        # We only consider gaze and video
        if sensor.type not in SENSOR_TYPES:
            continue

        # Fetch recent sensor configuration changes,
        # required for pyndsi internals
        while sensor.has_notifications:
            sensor.handle_notification()

        # Fetch recent gaze data
        for data in sensor.fetch_data():
            if data is None:
                continue
                    
            if sensor.name == "PI world v1":
                world_img = data.bgr
                #return world_img, -1

            elif sensor.name == "Gaze":
                # Draw gaze overlay onto world video frame
                gaze = (int(data[0]), int(data[1]))
                #return -1, gaze
                
    return world_img, gaze

