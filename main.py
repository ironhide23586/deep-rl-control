import pprint

import airsim
import numpy as np
import cv2


if __name__ == '__main__':
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)

    client.armDisarm(True)

    state = client.getMultirotorState()
    s = pprint.pformat(state)
    print("state: %s" % s)

    imu_data = client.getImuData()
    s = pprint.pformat(imu_data)
    print("imu_data: %s" % s)

    barometer_data = client.getBarometerData()
    s = pprint.pformat(barometer_data)
    print("barometer_data: %s" % s)

    magnetometer_data = client.getMagnetometerData()
    s = pprint.pformat(magnetometer_data)
    print("magnetometer_data: %s" % s)

    gps_data = client.getGpsData()
    s = pprint.pformat(gps_data)
    print("gps_data: %s" % s)

    landed = client.getMultirotorState().landed_state
    if landed == airsim.LandedState.Landed:
        print("taking off...")
        client.takeoffAsync().join()
    else:
        print("already flying...")
        client.hoverAsync().join()

    im_req = airsim.ImageRequest("1", airsim.ImageType.Scene, False, False)
    response = client.simGetImages([im_req])[0]
    im = np.frombuffer(response.image_data_uint8, dtype=np.uint8).reshape([response.height, response.width, 3])
    cv2.imwrite('im.png', im)

    client.moveByRollPitchYawThrottleAsync(0, np.deg2rad(45), 0, 0.5, 2.)
    client.moveByMotorPWMsAsync()

    k = 0
