#include <Wire.h>

// Pines del joystick
const int VRx = A0; 
const int VRy = A1;  

// Limites para determinar direcciones
const int threshold = 100;  
int xValue, yValue;  

void setup() {
  Wire.begin(8);  // Dirección I2C del esclavo
  Serial.begin(9600);
}

void loop() {
  // Leer los valores del joystick
  xValue = analogRead(VRx);  
  yValue = analogRead(VRy);  
  
  String command = getJoystickDirection(xValue, yValue);

  if (command != "STAY") {
    Wire.beginTransmission(8);  
    Wire.write(command.c_str());
    Wire.endTransmission();  
    Serial.println("Command sent: " + command);  
  }

  delay(200);  
}

// Función para determinar la dirección del joystick
String getJoystickDirection(int x, int y) {
  // Determinar dirección basándose en los valores leídos
  if (x < 512 - threshold) {
    return "LEFT";  // Izquierda
  } else if (x > 512 + threshold) {
    return "RIGHT";  // Derecha
  } else if (y < 512 - threshold) {
    return "FORWARD";  // Adelante
  } else if (y > 512 + threshold) {
    return "BACKWARD";  // Atrás
  }
  return "STAY";  // Quieto
}
