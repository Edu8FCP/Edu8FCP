"""
 *  This example is targeted toward the arduino platform
 *
 *  This example demonstrates the most simple usage of the Blue Robotics
 *  Ping1D c++ API in order to obtain distance and confidence reports from
 *  the device.
 *
 *  This API exposes the full functionality of the Ping1D Echosounder
 *
 *  https://github.com/bluerobotics/ping-arduino/blob/master/examples/ping1d-simple/ping1d-simple.ino
 *
 *  Communication is performed with a Blue Robotics Ping1D Echosounder
 *
 *  ADAPTADO PARA PYTHON (by Edu)
 *
 */
"""

#include "ping1d.h"

#include "SoftwareSerial.h"

# This serial port is used to communicate with the Ping device
# If you are using and Arduino UNO or Nano, this must be software serial, and you must use
# 9600 baud communication
# Here, we use pin 9 as arduino rx (Ping tx, white), 10 as arduino tx (Ping rx, green)
#static const uint8_t arduinoRxPin = 9;
#static const uint8_t arduinoTxPin = 10;
#SoftwareSerial pingSerial = SoftwareSerial(arduinoRxPin, arduinoTxPin);
#static Ping1D ping { pingSerial };

# static const uint8_t ledPin = 13;

void setup()
{
    pingSerial.begin(9600);
    Serial.begin(115200);
    pinMode(ledPin, OUTPUT);
    Serial.println("Blue Robotics ping1d-simple.ino");
    while (!ping.initialize()) {
        Serial.println("\nPing device failed to initialize!");
        Serial.println("Are the Ping rx/tx wired correctly?");
        Serial.print("Ping rx is the green wire, and should be connected to Arduino pin ");
        Serial.print(arduinoTxPin);
        Serial.println(" (Arduino tx)");
        Serial.print("Ping tx is the white wire, and should be connected to Arduino pin ");
        Serial.print(arduinoRxPin);
        Serial.println(" (Arduino rx)");
        delay(2000);
    }
}

def estado(): #void loop()
    if (ping.update()) {
        print("Distance: ");
        print(ping.distance());
        print("\tConfidence: ");
        print(ping.confidence());
    } else {
        print("No update received!");
    }

    # Toggle the LED to show that the program is running
    # digitalWrite(ledPin, !digitalRead(ledPin));
}