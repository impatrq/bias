#include "pico/stdlib.h"
#include "hardware/i2c.h"

#define I2C_SLAVE_ADDRESS 0x04

void i2c_slave_handler(i2c_inst_t *i2c, uint8_t *data, uint32_t len) {
    for (uint32_t i = 0; i < len; i++) {
        printf("Received data: %d\n", data[i]);
    }
}

int main() {
    stdio_init_all();

    // Initialize I2C as slave
    i2c_init(i2c0, 100000);
    gpio_set_function(4, GPIO_FUNC_I2C);
    gpio_set_function(5, GPIO_FUNC_I2C);
    gpio_pull_up(4);
    gpio_pull_up(5);

    i2c_slave_init(i2c0, I2C_SLAVE_ADDRESS, &i2c_slave_handler);

    while (true) {
        tight_loop_contents();  // Loop indefinitely
    }

    return 0;
}

void i2c_slave_init(i2c_inst_t *i2c, uint8_t address, void (*handler)(i2c_inst_t*, uint8_t*, uint32_t)) {
    i2c_set_slave_mode(i2c, true, address);
    i2c_set_irq_handler(i2c, I2C_SLAVE_IRQ, handler);
    i2c_set_irq_enabled(i2c, I2C_SLAVE_IRQ, true);
}