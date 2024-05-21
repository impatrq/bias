from machine import Pin, UART
from utime import sleep_ms

end = False

def main():
    uart = UART(0, baudrate=38400, tx=Pin(0), rx=Pin(1))

    enable = Pin(4, Pin.OUT)
    
    
    # uart = UART(0, baudrate=38400, tx=Pin(0), rx=Pin(1))
    
    # Dirección MAC (reemplázala con la real)
    #mac_adress = "8019,70,98695d"
    # mac_adress = "bc10,7b,35189f"
    # mac_adress = "ac5a,fc,056d9b"
    uart.write("AT\r\n".encode())
    #sleep_ms(1000)
    #print(uart.read().decode())
    
    mac_adress = "0081,F9,29EB31"
    
    
    connect_device(mac_adress, uart)
    
    #sleep_ms(5000)
    
    verify_connection(uart, mac_adress)
    
    enable.off()
    
    if end == True:
        uart.baudrate = 57600
    
    
    while True:
        if uart.any(): #Returns 0 if there are no characters available
            data = uart.read()
            print(data)
        '''
        else:
            print("No llega nada")
            sleep_ms(2000)
        '''
        
        '''
        if ser.any():
            uart.write(ser.read())
            
        if uart.any():
            ser.write(uart.read())
        '''
        
        
def connect_device(mac_adress, uart):
    
    uart.write("AT\r\n".encode())
    sleep_ms(1000)
    print(uart.read().encode())
    
    # Entra en modo comando AT
    uart.write("AT+NAME=HC\r\n".encode())
    sleep_ms(1000)
    print(uart.read().decode())
    
    uart.write("AT+UART=57600,0,0\r\n".encode())
    sleep_ms(1000)
    print(uart.read().decode())
    
    '''
    uart.write("AT+CMODE=0\r\n")
    sleep_ms(1000)
    print(uart.read().decode())
    '''
       
    # Rol de master
    uart.write("AT+ROLE=0\r\n".encode())
    sleep_ms(1000)
    print(uart.read().decode())
    
    
    uart.write("AT+PSWD=0000\r\n".encode())
    sleep_ms(1000)
    print(uart.read().decode())
    

    # Configuración para emparejar a dispositivo
    uart.write("AT+CMODE=0\r\n".encode())
    sleep_ms(1000)
    print(uart.read().decode())
    
    
    # Envía el comando AT para conectar al dispositivo
    comando_at = "AT+BIND=" + mac_adress + "\r\n"
    uart.write(comando_at.encode())
    sleep_ms(1000)
    print(uart.read().decode())
    
    
    uart.write("AT+IAC=9E8B33\r\n".encode())
    sleep_ms(1000)
    print(uart.read().decode())
    
    uart.write("AT+CLASS=0\r\n".encode())
    sleep_ms(1000)
    print(uart.read().decode())
    
    uart.write("AT+INQM=1,9,48\r\n".encode())
    sleep_ms(1000)
    print(uart.read().decode())
    
    print("End of connection")
    
    '''
    # Intenta conectar al dispositivo
    comando_at = "AT+LINK=" + mac_adress + "\r\n"
    uart.write(comando_at.encode())
    sleep_ms(1000)
    print(uart.read().decode())
    '''
    
    
def verify_connection(uart, mac_adress):
    
    uart.write("AT\r\n".encode())
    sleep_ms(1000)
    print(uart.read().decode())
    
    uart.write("AT+NAME?\r\n".encode())
    sleep_ms(1000)
    print(uart.read().decode())
    
    uart.write("AT+UART?\r\n".encode())
    sleep_ms(1000)
    print(uart.read().decode())

    # Rol de master
    uart.write("AT+ROLE?\r\n".encode())
    sleep_ms(1000)
    print(uart.read().decode())
    
    uart.write("AT+PSWD?\r\n".encode())
    sleep_ms(1000)
    print(uart.read().decode())

    # Configuración para emparejar a dispositivo
    uart.write("AT+CMODE?\r\n".encode())
    sleep_ms(1000)
    print(uart.read().decode())
    
    '''
    # Envía el comando AT para conectar al dispositivo
    comando_at = "AT+PAIR=" + mac_adress + "\r\n"
    uart.write(comando_at.encode())
    sleep_ms(1000)
    print(uart.read().decode())
    '''

    # Envía el comando AT para conectar al dispositivo
    uart.write("AT+BIND?\r\n".encode())
    sleep_ms(1000)
    print(uart.read().decode())
    
    uart.write("AT+IAC?\r\n".encode())
    sleep_ms(1000)
    print(uart.read().decode())
    
    uart.write("AT+CLASS?\r\n".encode())
    sleep_ms(1000)
    print(uart.read().decode())
    
    uart.write("AT+INQM?\r\n".encode())
    sleep_ms(1000)
    print(uart.read().decode())
    
    print("End of verification")
    '''
    # Envía el comando AT para conectar al dispositivo
    comando_at = "AT+LINK=" + mac_adress + "\r\n"
    uart.write(comando_at.encode())
    sleep_ms(5000)
    print(uart.read())
    '''
    
    
    sleep_ms(15000)
    
    uart.write("AT+STATE?\r\n".encode())
    sleep_ms(1000)  # Ajusta el tiempo si es necesario
    print(uart.read().decode())
    
    end = True
    
    
    # uart.write("AT+RESET\r\n".encode())
    # print(uart.read())
    
if __name__ == "__main__":
    main()
    
'''
  *   AT+UART=57600,0,0
  *   AT+ROLE=1
  *   AT+PSWD="0000"
  *   AT+CMODE=0
  *   AT+BIND=<Address>  ( 1234,56,ABCDEF )  2068,9D,79D995
  *   AT+IAC=9E8B33
  *   AT+CLASS=0
  *   AT+INQM=1,9,48
'''