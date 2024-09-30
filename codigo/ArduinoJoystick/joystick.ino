// Pines para los motores
const int motorIzqIN1 = 3;
const int motorIzqIN2 = 4;
const int motorDerIN3 = 5;
const int motorDerIN4 = 6;

// Pines del joystick
const int joyX = A0;  // Eje X
const int joyY = A1;  // Eje Y

int xValue, yValue;
int motorSpeed = 255;

void setup() {

  pinMode(motorIzqIN1, OUTPUT);
  pinMode(motorIzqIN2, OUTPUT);
  pinMode(motorDerIN3, OUTPUT);
  pinMode(motorDerIN4, OUTPUT);

  // Inicializar comunicación serial
  Serial.begin(9600);
}

void loop() {

  xValue = analogRead(joyX);
  yValue = analogRead(joyY);

  // Mapear los valores del joystick de 0-1023 a -512 a 512
  int xMapped = map(xValue, 0, 1023, -512, 512);
  int yMapped = map(yValue, 0, 1023, -512, 512);

  // Movimiento hacia adelante
  if (yMapped > 200) {
    moverAdelante();
  }
  // Movimiento hacia atrás
  else if (yMapped < -200) {
    moverAtras();
  }
  // Movimiento hacia la izquierda
  else if (xMapped < -200) {
    girarIzquierda();
  }
  // Movimiento hacia la derecha
  else if (xMapped > 200) {
    girarDerecha();
  }
  // No mover motores si el joystick está centrado
  else {
    pararMotores();
  }

  // (Opcional) Imprimir los valores del joystick para depuración
  Serial.print("X: ");
  Serial.print(xMapped);
  Serial.print(" Y: ");
  Serial.println(yMapped);

  delay(100); // Pequeño retardo para evitar lecturas demasiado rápidas
}

// Función para mover ambos motores hacia adelante
void moverAdelante() {
  digitalWrite(motorIzqIN1, HIGH);
  digitalWrite(motorIzqIN2, LOW);
  digitalWrite(motorDerIN3, HIGH);
  digitalWrite(motorDerIN4, LOW);
}

// Función para mover ambos motores hacia atrás
void moverAtras() {
  digitalWrite(motorIzqIN1, LOW);
  digitalWrite(motorIzqIN2, HIGH);
  digitalWrite(motorDerIN3, LOW);
  digitalWrite(motorDerIN4, HIGH);
}

// Función para girar hacia la izquierda 
void girarIzquierda() {
  digitalWrite(motorIzqIN1, LOW);
  digitalWrite(motorIzqIN2, LOW);
  digitalWrite(motorDerIN3, HIGH);
  digitalWrite(motorDerIN4, LOW);
}

// Función para girar hacia la derecha 
void girarDerecha() {
  digitalWrite(motorIzqIN1, HIGH);
  digitalWrite(motorIzqIN2, LOW);
  digitalWrite(motorDerIN3, LOW);
  digitalWrite(motorDerIN4, LOW);
}

// Función para detener ambos motores
void pararMotores() {
  digitalWrite(motorIzqIN1, LOW);
  digitalWrite(motorIzqIN2, LOW);
  digitalWrite(motorDerIN3, LOW);
  digitalWrite(motorDerIN4, LOW);
}
