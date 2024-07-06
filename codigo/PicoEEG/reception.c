#include "pico/stdlib.h"
#include "hardware/adc.h"
#include "hardware/i2c.h"
#include <stdio.h>

#define I2C_ADDR 0x3E
#define ADC_DELAY_US 1000
#define I2C_PORT i2c0
#define NUMBER_OF_BYTES_TO_READ 3
#define NUMBER_OF_CHANNELS 4
#define NUMBER_OF_BYTES_TO_SEND 8

bool read_adc();
void init_adc();

const int BAUDRATE = 100 * 1000;

const uint8_t SDA_PIN = 4;
const uint8_t SCL_PIN = 5;

const uint8_t ADC_PIN_CHANNEL_0 = 26;
const uint8_t ADC_PIN_CHANNEL_1 = 27;
const uint8_t ADC_PIN_CHANNEL_2 = 28;
const uint8_t ADC_PIN_CHANNEL_3 = 29;

// Whch pin to monitor - default to 0
static int ADC_CHANNEL = 0;

// ADC value
uint16_t adc_values[NUMBER_OF_CHANNELS];

// conversion factor is into mili volts
const float conversion_factor = 3.3f * 1000 / (1 << 12);


int main() {
    // setup ADC
    stdio_init_all();
    init_adc();

    // i2c setup
    i2c_init(I2C_PORT, BAUDRATE);
    i2c_set_slave_mode(I2C_PORT, true, I2C_ADDR);
    gpio_set_function(SDA_PIN, GPIO_FUNC_I2C);
    gpio_set_function(SCL_PIN, GPIO_FUNC_I2C);
    gpio_pull_up(SDA_PIN);
    gpio_pull_up(SCL_PIN);
    
    uint8_t rxdata[NUMBER_OF_BYTES_TO_READ + 1];
    uint8_t txdata[NUMBER_OF_BYTES_TO_SEND];
    //char message[20];

    int values[NUMBER_OF_CHANNELS];

    struct repeating_timer timer;
    add_repeating_timer_us(ADC_DELAY_US, read_adc, NULL, &timer);

    while (true) {
        // read_adc();
        for (int channel = 0; channel < NUMBER_OF_CHANNELS; channel++) {
            values[channel] = (int) adc_values[channel] * conversion_factor;
        }
        
        printf("adc_value0: %d; ", values[0]);
        printf("adc_value1: %d; ", values[1]);
        printf("adc_value2: %d; ", values[2]);
        printf("adc_value3: %d; ", values[3]);

        // Receive data from controller
        // 3 bytes received - byte 0 is cmd (used as lower byte) byte 2 is higher - byte 3 is 0
        if (i2c_get_read_available(I2C_PORT) < NUMBER_OF_BYTES_TO_READ) {
            continue;
        }

        i2c_read_raw_blocking(I2C_PORT, rxdata, NUMBER_OF_BYTES_TO_READ);
        //sprintf (message, "Rx: %d %d %d\r\n", rxdata[0], rxdata[1], rxdata[2]);
        // sprintf (message, "Value %d\r\n", rxdata[0] + (rxdata[1] << 8));

        printf("\nadc_value_sent_0: %d\n", values[0]);
        printf("\nadc_value_sent_1: %d\n", values[1]);
        printf("\nadc_value_sent_2: %d\n", values[2]);
        printf("\nadc_value_sent_3: %d\n", values[3]);
        
        //sleep_us(100);

        // Note that this will drop fraction rather than round, but close enough
        //int value = (int) adc_value * conversion_factor;
        for (int channel = 0, bit_number = 0; channel < NUMBER_OF_CHANNELS; channel++, bit_number++) {
            txdata[bit_number] = values[channel] & 0xFF;
            bit_number++;
            txdata[bit_number] = values[channel] >> 8;
        }
        /*
        txdata[0] = values[0] & 0xFF;
        txdata[1] = values[0] >> 8;
        txdata[2] = values[1] & 0xFF;
        txdata[3] = values[1] >> 8;
        txdata[4] = values[2] & 0xFF;
        txdata[5] = values[2] >> 8;
        txdata[6] = values[3] & 0xFF;
        txdata[7] = values[3] >> 8;
        */


        //sprintf(message, "Tx: %d %d - %d\r\n", txdata[0], txdata[1], values);

        // Sends data in mv (as int)
        i2c_write_raw_blocking(I2C_PORT, txdata, NUMBER_OF_BYTES_TO_SEND);
        // sleep_ms(10);
    }
}

bool read_adc() {
    for (int channel = 0; channel < NUMBER_OF_CHANNELS; channel++) {
        adc_select_input(channel);
        adc_values[channel] = adc_read();
        // printf("adc_values[%d]: %d\n", i, adc_values[i]);
    }
}

void init_adc() {
    adc_init();
    adc_gpio_init(ADC_PIN_CHANNEL_0); // ADC Channel 0 (GPIO 26)
    adc_gpio_init(ADC_PIN_CHANNEL_1); // ADC Channel 1 (GPIO 27)
    adc_gpio_init(ADC_PIN_CHANNEL_2); // ADC Channel 2 (GPIO 28)
    adc_gpio_init(ADC_PIN_CHANNEL_3); // ADC Channel 3 (GPIO 29)
    adc_select_input(ADC_CHANNEL);

    printf("ADC initialized\n");
}

/*
#include <stdio.h>
#include "pico/stdlib.h"
#include "hardware/adc.h"
#include "hardware/i2c.h"

#define I2C_PORT i2c0
#define I2C_ADDR 0x40 // Adjust the address as needed

uint16_t adc_values[4];

void init_adc() {
    adc_init();
    adc_gpio_init(26); // ADC Channel 0 (GPIO 26)
    adc_gpio_init(27); // ADC Channel 1 (GPIO 27)
    adc_gpio_init(28); // ADC Channel 2 (GPIO 28)
    adc_gpio_init(29); // ADC Channel 3 (GPIO 29)
    adc_select_input(0);

    printf("ADC initialized\n");
}

int main() {
    stdio_init_all();
    init_adc();

    i2c_init(I2C_PORT, 100 * 1000);
    gpio_set_function(4, GPIO_FUNC_I2C);
    gpio_set_function(5, GPIO_FUNC_I2C);
    gpio_pull_up(4);
    gpio_pull_up(5);

    i2c_set_slave_mode(I2C_PORT, true, I2C_ADDR);
    
    printf("I2C initialized as slave\n");

    while (true) {
        read_adc();

        uint8_t rxdata;
        int num_bytes = i2c_read_blocking(I2C_PORT, I2C_ADDR, &rxdata, 1, false);
        printf("rxdata: %d\n", rxdata);
        printf("num_bytes_read: %d\n", num_bytes);

        if (num_bytes > 0) {
            if (rxdata == 0) {
                int written = i2c_write_blocking(I2C_PORT, I2C_ADDR, (uint8_t *)adc_values, sizeof(adc_values), false);
                printf("num_bytes_written: %d\n", written);
            }
        }

        sleep_ms(20); // Adjust sleep time to match desired sample rate
    }

    return 0;
}

/*
#include "pico/stdlib.h"
#include "hardware/i2c.h"
#include "hardware/adc.h"
#include "hardware/irq.h"
#include <stdio.h>

#define I2C_SLAVE_ADDRESS 0x04
#define I2C_PORT i2c0
#define BUFFER_SIZE 256  // Adjust size as needed
#define NUM_CHANNELS 4

volatile uint16_t adc_buffer[BUFFER_SIZE][NUM_CHANNELS];
volatile int buffer_start = 0;
volatile int buffer_end = 0;

void i2c_slave_init(i2c_inst_t *i2c, uint8_t address);
void i2c_slave_handler();
void initialize_pins_i2c(i2c_inst_t *i2c, uint8_t sda, uint8_t scl, int baudrate);
void initialize_i2c(i2c_inst_t *i2c, uint8_t address, uint8_t sda, uint8_t scl, int baudrate);
void initialize_adc(uint8_t adc_pin);
uint16_t read_adc_value(uint8_t adc_input);

int main() {
    const uint8_t ADC_PINS[NUM_CHANNELS] = {26, 27, 28, 29};
    const uint8_t SDA_PIN = 4;
    const uint8_t SCL_PIN = 5;
    int baudrate = 100000;

    stdio_init_all();
    sleep_ms(2000);

    initialize_i2c(I2C_PORT, I2C_SLAVE_ADDRESS, SDA_PIN, SCL_PIN, baudrate);

    adc_init();
    for (int i = 0; i < NUM_CHANNELS; i++) {
        initialize_adc(ADC_PINS[i]);
    }

    while (true) {
        for (int i = 0; i < NUM_CHANNELS; i++) {
            adc_buffer[buffer_end][i] = read_adc_value(i);
        }
        buffer_end = (buffer_end + 1) % BUFFER_SIZE;
        if (buffer_end == buffer_start) {
            buffer_start = (buffer_start + 1) % BUFFER_SIZE;  // Overwrite oldest data if buffer is full
        }
        sleep_us(1000);  // Adjust sampling rate as needed
    }

    return 0;
}

void i2c_slave_init(i2c_inst_t *i2c, uint8_t address) {
    i2c_set_slave_mode(i2c, true, address);
    irq_set_exclusive_handler(I2C0_IRQ, i2c_slave_handler);
    irq_set_enabled(I2C0_IRQ, true);
}

void i2c_slave_handler() {
    i2c_inst_t *i2c = I2C_PORT;
    uint32_t status = i2c_get_hw(i2c)->intr_stat;

    if (status & I2C_IC_INTR_STAT_R_RD_REQ_BITS) {
        for (int i = 0; i < NUM_CHANNELS; i++) {
            uint16_t value = adc_buffer[buffer_start][i];
            uint8_t response[2];
            response[0] = (value >> 8) & 0xFF;
            response[1] = value & 0xFF;
            i2c_write_blocking(I2C_PORT, I2C_SLAVE_ADDRESS, response, sizeof(response), false);
        }
        buffer_start = (buffer_start + 1) % BUFFER_SIZE;  // Move to the next sample
        i2c_get_hw(i2c)->clr_rd_req;
    }
}

void initialize_pins_i2c(i2c_inst_t *i2c, uint8_t sda, uint8_t scl, int baudrate) {
    i2c_init(i2c, baudrate);
    gpio_set_function(sda, GPIO_FUNC_I2C);
    gpio_set_function(scl, GPIO_FUNC_I2C);
    gpio_pull_up(sda);
    gpio_pull_up(scl);
}

void initialize_i2c(i2c_inst_t *i2c, uint8_t address, uint8_t sda, uint8_t scl, int baudrate) {
    initialize_pins_i2c(i2c, sda, scl, baudrate);
    i2c_slave_init(i2c, address);
}

void initialize_adc(uint8_t adc_pin) {
    adc_gpio_init(adc_pin);
}

uint16_t read_adc_value(uint8_t adc_input) {
    adc_select_input(adc_input);
    uint16_t adc_value = adc_read();
    printf("adc_value[{%d}]: %d", adc_input, adc_value);
    return adc_value;
}
*/


/*
#include "pico/stdlib.h"
#include "hardware/i2c.h"
#include "hardware/irq.h"
#include "hardware/adc.h"
#include <stdio.h>

#define I2C_SLAVE_ADDRESS 0x04
#define I2C_PORT i2c0
#define BUFFER_SIZE 256
#define NUM_CHANNELS 4

uint16_t adc_buffer[BUFFER_SIZE][NUM_CHANNELS];
volatile uint16_t buffer_index = 0;

void i2c_slave_init(i2c_inst_t *i2c, uint8_t address);
void i2c_slave_handler();
void initialize_pins_i2c(i2c_inst_t *i2c, uint8_t sda, uint8_t scl, int baudrate);
void initialize_i2c(i2c_inst_t *i2c, uint8_t address, uint8_t sda, uint8_t scl, int baudrate);
void initialize_adc(uint8_t adc_pin);
uint16_t read_adc_value(uint8_t adc_input);
void core1_entry();

int main() {
    // Set ADC pins
    const uint8_t ADC_PINS[] = {26, 27, 28, 29};

    // Set i2c pins
    const uint8_t SDA_PIN = 4;
    const uint8_t SCL_PIN = 5;

    int baudrate = 100000;

    stdio_init_all();

    // Initialize I2C
    initialize_i2c(I2C_PORT, I2C_SLAVE_ADDRESS, SDA_PIN, SCL_PIN, baudrate);

    // Initialize ADC pins
    adc_init();
    for (int i = 0; i < NUM_CHANNELS; i++) {
        initialize_adc(ADC_PINS[i]);
    }

    multicore_launch_core1(core1_entry);

    while (true) {
        tight_loop_contents();
    }

    return 0;
}

void i2c_slave_init(i2c_inst_t *i2c, uint8_t address) {
    i2c_set_slave_mode(i2c, true, address);
    irq_set_exclusive_handler(I2C0_IRQ, i2c_slave_handler);
    irq_set_enabled(I2C0_IRQ, true);
}

void i2c_slave_handler() {
    i2c_inst_t *i2c = I2C_PORT;
    uint32_t status = i2c_get_hw(i2c)->intr_stat;

    if (status & I2C_IC_INTR_STAT_R_RD_REQ_BITS) {
        static uint16_t read_index = 0;
        for (int i = 0; i < NUM_CHANNELS; i++) {
            uint16_t adc_value = adc_buffer[read_index][i];
            i2c_get_hw(i2c)->data_cmd = (adc_value >> 8) & 0xFF;
            i2c_get_hw(i2c)->data_cmd = adc_value & 0xFF;
        }
        read_index = (read_index + 1) % BUFFER_SIZE;
        i2c_get_hw(i2c)->clr_rd_req;
    }
}

void initialize_pins_i2c(i2c_inst_t *i2c, uint8_t sda, uint8_t scl, int baudrate) {
    i2c_init(i2c, baudrate);
    gpio_set_function(sda, GPIO_FUNC_I2C);
    gpio_set_function(scl, GPIO_FUNC_I2C);
    gpio_pull_up(sda);
    gpio_pull_up(scl);
}

void initialize_i2c(i2c_inst_t *i2c, uint8_t address, uint8_t sda, uint8_t scl, int baudrate) {
    initialize_pins_i2c(i2c, sda, scl, baudrate);
    i2c_slave_init(i2c, address);
}

void initialize_adc(uint8_t adc_pin) {
    adc_gpio_init(adc_pin);
}

uint16_t read_adc_value(uint8_t adc_input) {
    adc_select_input(adc_input);
    return adc_read();
}

void core1_entry() {
    while (true) {
        for (int i = 0; i < NUM_CHANNELS; i++) {
            adc_buffer[buffer_index][i] = read_adc_value(i);
        }
        buffer_index = (buffer_index + 1) % BUFFER_SIZE;
        sleep_ms(1);  // Adjust this delay as needed to control the sampling rate
    }
}
*/

/*
#include "pico/stdlib.h"
#include "hardware/i2c.h"
#include "hardware/adc.h"
#include <stdio.h>
#include "pico/multicore.h"

#define I2C_SLAVE_ADDRESS 0x04
#define I2C_PORT i2c0

#define NUM_CHANNELS 4
#define PACKET_SIZE (NUM_CHANNELS * 2)  // 4 channels, 2 bytes per channel

volatile uint16_t adc_values[NUM_CHANNELS];
volatile uint8_t adc_packet[PACKET_SIZE];
volatile bool packet_ready = false;

void i2c_slave_handler();

void i2c_slave_init(i2c_inst_t *i2c, uint8_t address) {
    // Set I2C functions for the GPIO pins
    gpio_set_function(4, GPIO_FUNC_I2C);
    gpio_set_function(5, GPIO_FUNC_I2C);
    gpio_pull_up(4);
    gpio_pull_up(5);

    // Initialize I2C at the specified baudrate
    i2c_init(i2c, 100000);

    // Set the I2C to slave mode
    i2c_set_slave_mode(i2c, true, address);

    // Set up the interrupt handler for I2C
    irq_set_exclusive_handler(I2C0_IRQ, i2c_slave_handler);
    irq_set_enabled(I2C0_IRQ, true);
}

void initialize_adc(uint8_t adc_pin) {
    adc_gpio_init(adc_pin);
}

uint16_t read_adc_value(uint8_t adc_input) {
    adc_select_input(adc_input);
    float adc_value = adc_read();
    printf("adc_value: %f\n", adc_value);
    return adc_value;
}

void create_adc_packet() {
    for (int i = 0; i < NUM_CHANNELS; i++) {
        adc_values[i] = read_adc_value(i);  // Read from ADC channel i
        adc_packet[i * 2] = (adc_values[i] >> 8) & 0xFF;  // High byte
        adc_packet[i * 2 + 1] = adc_values[i] & 0xFF;     // Low byte
    }
    packet_ready = true;
}

void adc_reading_task() {
    while (true) {
        create_adc_packet();
        sleep_ms(2);  // Adjust the sleep time as necessary to match the desired sampling rate
    }
}

void i2c_slave_handler() {
    i2c_inst_t *i2c = I2C_PORT;
    uint32_t status = i2c_get_hw(i2c)->intr_stat;

    static uint8_t packet_index = 0;

    if (status & I2C_IC_INTR_STAT_R_RD_REQ_BITS) {
        if (packet_ready) {
            i2c_get_hw(i2c)->data_cmd = adc_packet[packet_index];
            packet_index++;
            if (packet_index >= PACKET_SIZE) {
                packet_index = 0;
                packet_ready = false;  // Packet has been fully sent, prepare for the next packet
            }
        }
        i2c_get_hw(i2c)->clr_rd_req;
    }
}

int main(void) {
    stdio_init_all();

    // Initialize ADC channels
    for (uint8_t i = 26; i < 30; i++) {
        initialize_adc(i);
    }

    i2c_slave_init(I2C_PORT, I2C_SLAVE_ADDRESS);

    // Launch the ADC reading task
    multicore_launch_core1(adc_reading_task);

    while (true) {
        tight_loop_contents();  // Loop indefinitely
    }

    return 0;
}
*/
/*
#include "pico/stdlib.h"
#include "hardware/i2c.h"
#include "hardware/irq.h"
#include "hardware/adc.h"
#include <stdio.h>

#define I2C_SLAVE_ADDRESS 0x04
#define I2C_PORT i2c0

uint16_t adc_values[4];

void init_adc() {
    adc_init();
    adc_gpio_init(26); // ADC Channel 0 (GPIO 26)
    adc_gpio_init(27); // ADC Channel 1 (GPIO 27)
    adc_gpio_init(28); // ADC Channel 2 (GPIO 28)
    adc_gpio_init(29); // ADC Channel 3 (GPIO 29)
    adc_select_input(0);
}

void read_adc() {
    for (int i = 0; i < 4; i++) {
        adc_select_input(i);
        adc_values[i] = adc_read();
    }
}

void i2c_slave_callback(i2c_inst_t *i2c, i2c_slave_event_t event) {
    static int byte_count = 0;

    switch (event) {
        case I2C_SLAVE_RECEIVE:
            // Receive data from master (if needed)
            break;
        case I2C_SLAVE_REQUEST:
            // Master is requesting data
            i2c_write_raw_blocking(I2C_PORT, (uint8_t *)adc_values, sizeof(adc_values));
            break;
        case I2C_SLAVE_FINISH:
            byte_count = 0;
            break;
    }
}

int main() {
    stdio_init_all();
    init_adc();

    i2c_init(I2C_PORT, 100 * 1000);
    gpio_set_function(4, GPIO_FUNC_I2C);
    gpio_set_function(5, GPIO_FUNC_I2C);
    gpio_pull_up(4);
    gpio_pull_up(5);

    i2c_slave_init(I2C_PORT, I2C_SLAVE_ADDRESS, &i2c_slave_callback);

    while (true) {
        read_adc();
        sleep_ms(2); // Adjust sleep time to match desired sample rate
    }

    return 0;
}
*/

/*
void i2c_slave_init(i2c_inst_t *i2c, uint8_t address);
void i2c_slave_handler();
void initialize_pins_i2c(i2c_inst_t *i2c, uint8_t sda, uint8_t scl, int baudrate);
void initialize_i2c(i2c_inst_t *i2c, uint8_t address, uint8_t sda, uint8_t scl, int baudrate);
void initialize_adc(uint8_t adc_pin);
uint16_t read_adc_value(uint8_t adc_input);

int main() {
    // Set ADC pins and channels
    const uint8_t ADC_PIN_1 = 26;
    const uint8_t ADC_PIN_2 = 27;
    const uint8_t ADC_PIN_3 = 28;
    const uint8_t ADC_PIN_4 = 29;

    // Set i2c pins
    const uint8_t SDA_PIN = 4;
    const uint8_t SCL_PIN = 5;

    int baudrate = 100000;

    stdio_init_all();

    sleep_ms(2000);

    // Initialize I2C
    initialize_i2c(I2C_PORT, I2C_SLAVE_ADDRESS, SDA_PIN, SCL_PIN, baudrate);

    // Initialize ADC pins and channels
    adc_init();
    initialize_adc(ADC_PIN_1);
    initialize_adc(ADC_PIN_2);
    initialize_adc(ADC_PIN_3);
    initialize_adc(ADC_PIN_4);

    while (true) {
        tight_loop_contents();  // Loop indefinitely
    }

    return 0;
}

// Set Raspberry Pi Pico as slave
void i2c_slave_init(i2c_inst_t *i2c, uint8_t address) {
    i2c_set_slave_mode(i2c, true, address);
    printf("Slave mode initialized correctly");
    // Handle interruptions
    irq_set_exclusive_handler(I2C0_IRQ, i2c_slave_handler);
    printf("No problem with irq_set_exclusive_handler");
    irq_set_enabled(I2C0_IRQ, true);
    printf("No problem with irq_set_enabled");
}

void i2c_slave_handler() {
    // Assuming i2c0
    i2c_inst_t *i2c = I2C_PORT;

    uint32_t status = i2c_get_hw(i2c)->intr_stat;

    // Declare variables for ADC reading
    static uint8_t adc_input = 0;
    uint16_t adc_value;

    if (status & I2C_IC_INTR_STAT_R_RX_FULL_BITS) {
        uint8_t data = i2c_get_hw(i2c)->data_cmd;
        printf("Received data: %d\n", data);
        // Use received data to select ADC input, assuming data is valid ADC input channel
        adc_input = data;
        
        // Clear the RX interrupt
        i2c_get_hw(i2c)->clr_rx_under;
        i2c_get_hw(i2c)->clr_rx_over;
        i2c_get_hw(i2c)->clr_rx_done;
    }

    //printf("I2C_IC_INTR_STAT_R_RD_REQ_BITS: %d", I2C_IC_INTR_STAT_R_RD_REQ_BITS);
    if (status & I2C_IC_INTR_STAT_R_RD_REQ_BITS) {
        // Read the selected ADC input
        adc_value = read_adc_value(adc_input);
        printf("adc_value: %d", adc_value);
        // Read the two bytes since ADC has 16 bits
        uint8_t response[2];
        // Join the two parts of the response
        response[0] = (adc_value >> 8) & 0xFF;
        //printf("response[0]: %d", response[0]);
        response[1] = adc_value & 0xFF;
        //printf("response[1]: %d", response[1]);
        printf("Response: [%d, %d]\n", response[0], response[1]);
        i2c_get_hw(i2c)->data_cmd = response[0];
        i2c_get_hw(i2c)->data_cmd = response[1];
        i2c_get_hw(i2c)->clr_rd_req;
    }
}

void initialize_pins_i2c(i2c_inst_t *i2c, uint8_t sda, uint8_t scl, int baudrate) {
    
    // Initialize I2C as slave
    i2c_init(i2c, baudrate);
    printf("I2c init correctly");

    // Set SDA and SCL pins
    gpio_set_function(sda, GPIO_FUNC_I2C);
    gpio_set_function(scl, GPIO_FUNC_I2C);
    gpio_pull_up(sda);
    gpio_pull_up(scl);
    printf("Pins correct");
}

void initialize_i2c(i2c_inst_t *i2c, uint8_t address, uint8_t sda, uint8_t scl, int baudrate) {
    // Initialize I2C
    initialize_pins_i2c(i2c, sda, scl, baudrate);
    printf("Pins initialized correctly");
    i2c_slave_init(i2c, address);
    printf("i2c_slave_init correctly");
}

void initialize_adc(uint8_t adc_pin) {
    // Initialize ADC
    adc_gpio_init(adc_pin);
}

uint16_t read_adc_value(uint8_t adc_input) {
    // Read specific ADC channel
    adc_select_input(adc_input);
    return adc_read();
    for (int i = 0; i < 4; i++) {
        adc_select_input(i);
        adc_values[i] = adc_read();
    }
}
*/