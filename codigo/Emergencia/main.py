from machine import Pin, ADC
from time import sleep, sleep_us, ticks_us
from math import atan2, degrees

# Establecer variables globales para ADC
NUM_STEPS = 2 ** 16
MAX_RADIUS = (NUM_STEPS - 1) / 2
TOLERANCE_STEP_IN_REST = 7500
    
def main():
    # Setear los pines de ADC
    pinx = ADC(Pin(26))
    piny = ADC(Pin(27))
    
    # Setear los ultrasonidos
    trig_forward = Pin(16, Pin.OUT)
    echo_forward = Pin(15, Pin.IN, Pin.PULL_DOWN)
    trig_back = Pin(14, Pin.OUT)
    echo_back = Pin(13, Pin.IN, Pin.PULL_DOWN)
    trig_right = Pin(12, Pin.OUT)
    echo_right = Pin(11, Pin.IN, Pin.PULL_DOWN)
    trig_left = Pin(10, Pin.OUT)
    echo_left = Pin(9, Pin.IN, Pin.PULL_DOWN)
    
    # Setear los pines de los motores
    motor_left_forward = Pin(0, Pin.OUT)
    motor_left_reverse = Pin(1, Pin.OUT)
    motor_right_forward = Pin(2, Pin.OUT)
    motor_right_reverse = Pin(3, Pin.OUT)
    
    # Setear el sistema de emergencia
    led_forward = Pin(4, Pin.OUT)
    led_behind = Pin(5, Pin.OUT)
    led_left = Pin(6, Pin.OUT)
    led_right = Pin(7, Pin.OUT)
    buzzer = Pin(8, Pin.OUT)
    
    while True: 
        is_object_forward = detected_object(trig_forward, echo_forward)
        is_object_back = detected_object(trig_back, echo_back)
        is_object_left = detected_object(trig_left, echo_left)
        is_object_right = detected_object(trig_right, echo_right)
        
        # Lee donde el joystick está apuntando
        direction = read_joystick(pinx, piny)
        
        not_possible_directions = not_allowed_directions(is_object_forward, is_object_back, is_object_left, is_object_right)
        
        for i in not_possible_directions:
            if direction == i:
                emergency_system(direction, led_forward, led_right, led_left, led_back, buzzer)
                direction = "Parar"
                sleep(1)
                break
        
        sleep(0.1)
        
        move_motors(direction, motor_left_forward, motor_left_reverse, motor_right_forward, motor_right_reverse)

def read_joystick(pinx, piny):
    # Lee el ADC
    xvalue = pinx.read_u16() - MAX_RADIUS
    yvalue = - piny.read_u16() + MAX_RADIUS
    print(f"X: {xvalue}")
    print(f"Y: {yvalue}")
    
    if abs(xvalue) < TOLERANCE_STEP_IN_REST and abs(yvalue) < TOLERANCE_STEP_IN_REST:
        print("Reposo")
        return "Reposo"
    
    angle = degrees(atan2(yvalue, xvalue))
    print(f"Angle: {angle}")
    
    if (angle > -180 and angle <= -157.5) or (angle >= 157.5 and angle <= 180):
        print("Izquierda")
        return "Izquierda"
    elif (angle >= 0 and angle <= 22.5) or (angle < 0 and angle >= -22.5):
        print("Derecha")
        return "Derecha"
    elif angle <= -67.5 and angle >= -112.5:
        print("Atras")
        return "Atras"
    elif angle >= 67.5 and angle <= 112.5:
        print("Adelante")
        return "Adelante"  
    elif angle < -22.5 and angle > -67.5:
        print("Atras y derecha")
        return "Atras y derecha"
    elif angle < -112.5 and angle > -157.5:
        print("Atras e izquierda")
        return "Atras e izquierda"
    elif angle < 67.5 and angle > 22.5:
        print("Adelante y derecha")
        return "Adelante y derecha"
    elif angle < 157.5 and angle > 112.5:
        print("Adelante e izquierda")
        return "Adelante e izquierda"
    else:
        print("Reposo")
        return "Reposo"

# Mueve los motores según la orden del joystick
def move_motors(direction, motor_left_forward, motor_left_reverse, motor_right_forward, motor_right_reverse):
    if direction == "Reposo":
        stop(motor_left_forward, motor_left_reverse, motor_right_forward, motor_right_reverse)
    elif direction == "Adelante":
        forward(motor_left_forward, motor_left_reverse, motor_right_forward, motor_right_reverse)
    elif direction == "Atras":
        back(motor_left_forward, motor_left_reverse, motor_right_forward, motor_right_reverse)
    elif direction == "Izquierda":
        left(motor_left_forward, motor_left_reverse, motor_right_forward, motor_right_reverse)
    elif direction == "Derecha":
        right(motor_left_forward, motor_left_reverse, motor_right_forward, motor_right_reverse)
    elif direction == "Adelante y derecha":
        forward_right(motor_left_forward, motor_left_reverse, motor_right_forward, motor_right_reverse)
    elif direction == "Adelante e izquierda":
        forward_left(motor_left_forward, motor_left_reverse, motor_right_forward, motor_right_reverse)
    elif direction == "Atras e izquierda":
        back_left(motor_left_forward, motor_left_reverse, motor_right_forward, motor_right_reverse)
    elif direction == "Atras y derecha":
        back_right(motor_left_forward, motor_left_reverse, motor_right_forward, motor_right_reverse)
    elif direction == "Parar":
        stop(motor_left_forward, motor_left_reverse, motor_right_forward, motor_right_reverse)
    else:
        stop(motor_left_forward, motor_left_reverse, motor_right_forward, motor_right_reverse)

def stop(motor_left_forward, motor_left_reverse, motor_right_forward, motor_right_reverse):
    motor_left_forward.off()
    motor_left_reverse.off()
    motor_right_forward.off()
    motor_right_reverse.off()
    
    
def forward(motor_left_forward, motor_left_reverse, motor_right_forward, motor_right_reverse):
    motor_left_forward.on()
    motor_left_reverse.off()
    motor_right_forward.on()
    motor_right_reverse.off()


def right(motor_left_forward, motor_left_reverse, motor_right_forward, motor_right_reverse):
    motor_left_forward.on()
    motor_left_reverse.off()
    motor_right_forward.off()
    motor_right_reverse.on()
    
def left(motor_left_forward, motor_left_reverse, motor_right_forward, motor_right_reverse):
    motor_left_forward.off()
    motor_left_reverse.on()
    motor_right_forward.on()
    motor_right_reverse.off()

def back(motor_left_forward, motor_left_reverse, motor_right_forward, motor_right_reverse):
    motor_left_forward.off()
    motor_left_reverse.on()
    motor_right_forward.off()
    motor_right_reverse.on()

def forward_right(motor_left_forward, motor_left_reverse, motor_right_forward, motor_right_reverse):
    motor_left_forward.on()
    motor_left_reverse.off()
    motor_right_forward.off()
    motor_right_reverse.off()

def forward_left(motor_left_forward, motor_left_reverse, motor_right_forward, motor_right_reverse):
    motor_left_forward.off()
    motor_left_reverse.off()
    motor_right_forward.on()
    motor_right_reverse.off()

def back_right(motor_left_forward, motor_left_reverse, motor_right_forward, motor_right_reverse):
    motor_left_forward.off()
    motor_left_reverse.on()
    motor_right_forward.off()
    motor_right_reverse.off()
    
def back_left(motor_left_forward, motor_left_reverse, motor_right_forward, motor_right_reverse):
    motor_left_forward.off()
    motor_left_reverse.off()
    motor_right_forward.off()
    motor_right_reverse.on()
        
def detected_object(trig, echo):
    trig.value(0)
    sleep(0.1)
    trig.value(1)
    sleep_us(2)
    trig.value(0)
    while echo.value() == 0:
        pulse_start = ticks_us()
    while echo.value() == 1:
        pulse_end = ticks_us()
    pulse_duration = pulse_end - pulse_start
    distance = pulse_duration * 17165 / 1000000
    distance = round(distance, 0)
    print ('Distance:',"{:.0f}".format(distance),'cm')
    if distance < 25:
        print("Objeto detectado")
        return True
    sleep(1)
    return False
    
def not_allowed_directions(is_object_forward, is_object_back, is_object_left, is_object_right):
    not_allowed_directions = list()
    
    if is_object_forward == True:
        for direction in ["Adelante", "Adelante e izquierda", "Adelante y derecha"]:
            not_allowed_directions.append(direction)
        
    if is_object_back == True:
        for direction in ["Atras", "Atras e izquierda", "Atras y derecha"]:
            not_allowed_directions.append(direction)
        
    if is_object_right == True:
        not_allowed_directions.append("Derecha")
        
    if is_object_left == True:
        not_allowed_directions.append("Izquierda")
        
    return not_allowed_directions
        
def emergency_system(direction, led_forward, led_right, led_left, led_back, buzzer):
    if direction in ["Adelante", "Adelante e izquierda", "Adelante y derecha"]:
        led_forward.on()
    elif direction in ["Atras", "Atras e izquierda", "Atras y derecha"]
        led_back.on()
    elif direction == "Derecha":
        led_right.on()
    elif direction == "Izquierda":
        led_left.on()
    
    buzzer.on()
    sleep(1)

if __name__ == "__main__":
    main()