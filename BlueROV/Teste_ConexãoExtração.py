# import Ping360 class
from brping import ping360
import numpy as np

# Create Ping360 instance
sonar = Ping360()

if sonar.initialize() is False:
    print("Failed to initialize Ping!")
    exit(1)

#Use get_<message_name> no request data
data = sonar.get_distance()
if data:
    print("Distance: %s\tConfidence: %s%%" % (data["distance"], data["confidence"]))
else:
    print("Failed to get distance data")

# set the speed of sound to use for distance calculations to
    # 1450000 mm/s (1450 m/s)
    sonar.set_speed_of_sound(1450000)

#--------------------------------------------------------
#
# NEW TRY
#
#--------------------------------------------------------

#RECEIVING DATA



sonar.connect_udp("192.168.2.2", 9092) # Connect to, initialize, and set up Ping360 settings
#9092 Ping360 9090 1D

if sonar.initialize() is False:
    print("Failed to initialize Ping!")
    exit(1)
    #is True: 
    print("Success")
# Loop through a full circle, one gradian at a time
for x in range(400):
    response = sonar.transmitAngle(x)
    print(response)

#PROCESSING DATA

... # create and initialise Ping360 object

response = sonar.transmitAngle(x)
data = np.frombuffer(response.data, dtype=np.uint8)
print(data.min(), data.max())
# print all locations that are above threshold
threshold = 200
print(np.where(data >= threshold))