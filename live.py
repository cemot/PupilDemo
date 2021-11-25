import time

import cv2
import numpy as np

# https://github.com/pupil-labs/pyndsi/tree/v1.0
import ndsi  # Main requirement

from heatmappy import Heatmapper
from PIL import Image

SENSOR_TYPES = ["video", "gaze"]
SENSORS = {}  # Will store connected sensors


def main():
    # Start auto-discovery of Pupil Invisible Companion devices
    network = ndsi.Network(
        formats={ndsi.DataFormat.V4}, callbacks=(on_network_event,))
    network.start()

    try:
        #
        world_img = np.zeros((1088, 1080, 3))
        gaze = (0, 0)
        gazes = []
        
        cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("window",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
        cv2.imshow("window", world_img)
        key = cv2.waitKey(1)

        # Event loop, runs until interrupted
        while network.running:
            # Check for recently connected/disconnected devices
            if network.has_events:
                network.handle_event()

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

                    elif sensor.name == "Gaze":
                        # Draw gaze overlay onto world video frame
                        gaze = (int(data[0]), int(data[1]))
                        gazes.append(gaze)

            # Show world video with gaze overlay
            cv2.circle(
                world_img,
                gaze,
                40, (0, 0, 255), 4
            )
            cv2.imshow("window", world_img)
            key = cv2.waitKey(1)
            
            if key == ord("q"):
                #world_img = np.uint8(world_img)
                world_img = cv2.cvtColor(world_img, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(world_img)
                
                #cv2.imwrite('temp.jpg',world_img)
                #temp_img = Image.open("temp.jpg")
                heatmapper = Heatmapper()
                heatmap = heatmapper.heatmap_on_img(gazes, pil_img)
                heatmap = np.array(heatmap)
                
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)

                cv2.imshow("window", heatmap)
                key = cv2.waitKey(0)
                
                network.stop()
                return 1

    # Catch interruption and disconnect gracefully
    except (KeyboardInterrupt, SystemExit):
        network.stop()


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
        print(f"Added sensor {sensor}...")

    # Handle gaze sensor detachment
    if event["subject"] == "detach" and event["sensor_uuid"] in SENSORS:
        # Known sensor has disconnected, remove from list
        SENSORS[event["sensor_uuid"]].unlink()
        del SENSORS[event["sensor_uuid"]]
        print(f"Removed sensor {event['sensor_uuid']}...")


main()  # Execute example
cv2.destroyAllWindows()
