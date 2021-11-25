import time
import uuid
import json

# https://github.com/pupil-labs/pyndsi/
import ndsi  # Main requirement

HARDWARE_TYPE = "hardware"  # Type of sensors that we are interested in
SENSORS = {}  # Will store connected sensors

RECORDING_TIME = 5  # seconds

timestamps_recording_start = {}


def main():
    # Start auto-discovery of Pupil Invisible Companion devices
    network = ndsi.Network(formats={ndsi.DataFormat.V4}, callbacks=(on_network_event,))
    network.start()

    try:
        # Event loop, runs until interrupted
        while network.running:
            # Check for recently connected/disconnected devices
            if network.has_events:
                network.handle_event()

            # Iterate over all connected devices
            for hardware_sensor in SENSORS.values():
                # Fetch recent sensor configuration changes,
                # required for pyndsi internals
                while hardware_sensor.has_notifications:
                    hardware_sensor.handle_notification()

                stop_recording(hardware_sensor)
        time.sleep(0.1)

    # Catch interruption and disconnect gracefully
    except (KeyboardInterrupt, SystemExit):
        network.stop()


def start_recording(hardware_sensor):
    if hardware_sensor not in timestamps_recording_start:
        # start recording
        hardware_sensor.set_control_value("local_capture", True)
        # request the current configuration
        hardware_sensor.refresh_controls()
        timestamps_recording_start[hardware_sensor] = time.time()
        print(f"started recording for sensor {hardware_sensor.uuid}")


def stop_recording(hardware_sensor):
    # check if the sensor is recording
    if (
        hardware_sensor in timestamps_recording_start
        and timestamps_recording_start[hardware_sensor] is not None
    ):
        rec_time_start = timestamps_recording_start[hardware_sensor]
        now = time.time()
        if now - rec_time_start >= RECORDING_TIME:
            # stop the recording after 5 seconds recording time
            hardware_sensor.set_control_value("local_capture", False)
            # request the current configuration
            hardware_sensor.refresh_controls()
            timestamps_recording_start[hardware_sensor] = None
            print(f"stopped recording for sensor {hardware_sensor.uuid}")


def on_network_event(network, event):
    # Handle event sensor attachment
    if event["subject"] == "attach" and event["sensor_type"] == HARDWARE_TYPE:
        # Create new sensor
        # and request current configuration
        sensor = network.sensor(event["sensor_uuid"])
        sensor.refresh_controls()

        # Save sensor s.t. we can fetch data from it in main()
        SENSORS[event["sensor_uuid"]] = sensor
        print(f"Added sensor {sensor}...")

        # start the recording
        start_recording(sensor)


    # Handle event sensor detachment
    if event["subject"] == "detach" and event["sensor_uuid"] in SENSORS:
        # Known sensor has disconnected, remove from list
        SENSORS[event["sensor_uuid"]].unlink()
        del SENSORS[event["sensor_uuid"]]
        print(f"Removed sensor {event['sensor_uuid']}...")


if __name__ == "__main__":
    main()  # Execute example

