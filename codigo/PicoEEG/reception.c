#include "pico/stdlib.h"
#include "hardware/i2c.h"
#include "hardware/irq.h"
#include "hardware/adc.h"
#include <stdio.h>

#define I2C_SLAVE_ADDRESS 0x04
#define I2C_PORT i2c0

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
    printf("No roblem with irq_set_enabled");
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
        printf("response[0]: %d", response[0]);
        response[1] = adc_value & 0xFF;
        printf("response[1]: %d", response[1]);
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
}