#include "pico/stdlib.h"
#include "hardware/i2c.h"
#include "hardware/irq.h"
#include "hardware/adc.h"

#define I2C_SLAVE_ADDRESS 0x04
#define I2C_PORT i2c0

void i2c_slave_init(i2c_inst_t *i2c, uint8_t address);
void i2c_slave_handler(i2c_inst_t *i2c, uint32_t event_mask);
void initialize_pins_i2c(i2c_inst_t *i2c, uint8_t sda, uint8_t scl, int baudrate);
void initialize_i2c(i2c_inst_t *i2c, uint8_t address, uint8_t sda, uint8_t scl, int baudrate);
void initialize_adc(uint8_t adc_pin, uint8_t adc_input);

int main() {
    const uint8_t ADC_PIN_1 = 26;
    const uint8_t ADC_INPUT_1 = 0;
    const uint8_t ADC_PIN_2 = 27;
    const uint8_t ADC_INPUT_2 = 1;
    const uint8_t ADC_PIN_3 = 28;
    const uint8_t ADC_INPUT_3 = 2;
    const uint8_t ADC_PIN_4 = 29;
    const uint8_t ADC_INPUT_4 = 3;
    const uint8_t SDA_PIN = 4;
    const uint8_t SCL_PIN = 5;

    int baudrate = 100000;

    stdio_init_all();

    initialize_pins_i2c(I2C_PORT, SDA_PIN, SCL_PIN, baudrate);
    i2c_slave_init(I2C_PORT, I2C_SLAVE_ADDRESS);

    initialize_adc(ADC_PIN_1, ADC_INPUT_1);
    initialize_adc(ADC_PIN_2, ADC_INPUT_2);
    initialize_adc(ADC_PIN_3, ADC_INPUT_3);
    initialize_adc(ADC_PIN_4, ADC_INPUT_4);

    while (true) {
        tight_loop_contents();  // Loop indefinitely skibidi toilet
    }

    return 0;
}

void i2c_slave_init(i2c_inst_t *i2c, uint8_t address) {
    i2c_set_slave_mode(i2c, true, address);
    irq_set_exclusive_handler(I2C0_IRQ, i2c_slave_handler);
    irq_set_enabled(I2C0_IRQ, true);
}

void i2c_slave_handler(i2c_inst_t *i2c, uint32_t event_mask) {
    uint8_t data;
    if (event_mask & I2C_SLAVE_RECEIVE) {
        if (i2c_slave_get_data(i2c, &data, 1)) {
            printf("Received data: %d\n", data);
        }
    }
    if (event_mask & I2C_SLAVE_REQUEST) {
        // Respond with data if the master requests
        uint8_t response = 42;  // Example response skibidi toilet
        i2c_slave_write(i2c, &response, 1);
    }
}

void initialize_pins_i2c(i2c_inst_t *i2c, uint8_t sda, uint8_t scl, int baudrate) {
    // Initialize I2C as slave
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
void initialize_adc(uint8_t adc_pin, uint8_t adc_input) {
    adc_init();
    adc_gpio_init(adc_pin);
    adc_select_input(adc_input);
}