#include <atmel_start.h>
#include "driver_examples.h"
#include "driver_init.h"
#include "utils.h"

#define START_BYTE 0x02  // Example start byte
#define MAX_PACKET_LENGTH 256 // Define a maximum packet length to avoid overly large packets
volatile bool data_received = false;
uint8_t rx_buffer[MAX_PACKET_LENGTH];
uint16_t rx_index = 0;
bool packet_started = false;
uint8_t mode_byte;
uint8_t packet_length = 0;
volatile uint8_t interrupt_count = 0;
volatile bool triggout_state = false;

void set_XVS_freq(void)
{
	//PA15
	pwm_set_parameters(&PWM_0, 50000, 25000); // 30Hz in clock cycles for a 24MHz clock
	
}

void set_XHS_freq(void)
{
	//PA23
	pwm_set_parameters(&PWM_1, 178, 89); // 14.81us for 1H, half of it=7.4us, which is ~134kHz, so, 24MHz/178 is ~134kHz
	pwm_enable(&PWM_1);
}


void LED_blink_callback(void)
{
	if (gpio_get_pin_level(LED))
	gpio_set_pin_level(LED,false);
	else
	gpio_set_pin_level(LED,true);
	
}

static void tx_cb_USART_0(const struct usart_async_descriptor *const io_descr)
{
	
}

static void rx_cb_USART_0(const struct usart_async_descriptor *const io_descr)
{
	struct io_descriptor *io;
	usart_async_get_io_descriptor(&USART_0, &io);
	uint8_t received_byte;
	io_read(io, &received_byte, 1);
	
	if (!packet_started) {
		if (received_byte == START_BYTE) {
			packet_started = true;
			rx_index = 0;
			packet_length = 0; // Reset packet length
		}
		} else {
		rx_buffer[rx_index++] = received_byte;
		if (rx_index == 1) {
			packet_length = received_byte; // Set the packet length from the first byte after START_BYTE
			if (packet_length > MAX_PACKET_LENGTH) {
				packet_length = MAX_PACKET_LENGTH; // Limit the packet length to the maximum allowed
			}
			} else if (rx_index == 2) {
			mode_byte = received_byte;; // Set the mode byte from the second byte after START_BYTE
		}
		if (packet_length && rx_index >= packet_length) {
			data_received = true;
			packet_started = false;
			rx_index = 0;
		}
	}
}

void configure_usart(void)
{
	struct io_descriptor *io;
	usart_async_register_callback(&USART_0, USART_ASYNC_RXC_CB, rx_cb_USART_0);
	usart_async_register_callback(&USART_0, USART_ASYNC_TXC_CB, tx_cb_USART_0);
	usart_async_get_io_descriptor(&USART_0, &io);
	usart_async_enable(&USART_0);
}

void I2C_0_tx_complete(struct i2c_m_async_desc *const i2c)
{
}

void I2C_2_dlp_rgb_common(uint8_t *data, uint8_t length)
{
	struct io_descriptor *I2C_0_io;

	i2c_m_async_get_io_descriptor(&I2C_0, &I2C_0_io);
	i2c_m_async_enable(&I2C_0);
	i2c_m_async_register_callback(&I2C_0, I2C_M_ASYNC_TX_COMPLETE, (FUNC_PTR)I2C_0_tx_complete);
	i2c_m_async_set_slaveaddr(&I2C_0, 0x1B, I2C_M_SEVEN);

	io_write(I2C_0_io, data, length);
}

void I2C_2_dlp_read_common(uint8_t *data, uint8_t length)
{
	struct io_descriptor *I2C_0_io;

	i2c_m_async_get_io_descriptor(&I2C_0, &I2C_0_io);
	i2c_m_async_enable(&I2C_0);
	i2c_m_async_register_callback(&I2C_0, I2C_M_ASYNC_TX_COMPLETE, (FUNC_PTR)I2C_0_tx_complete);
	i2c_m_async_set_slaveaddr(&I2C_0, 0x1B, I2C_M_SEVEN);

	io_read(I2C_0_io, data, length);
}

uint8_t RVET_dataA[] = {0x9D, 0x00, 0x03, 0xF8, 0x2A, 0x00, 0x00};
//uint8_t R_H01_data[] = {0x01, 0xF8, 0x2A, 0x00, 0x00, 0xAC, 0x00, 0x00, 0x00, 0x1F, 0x00, 0x00, 0x00, 0x13};
uint8_t R_H01_dataA[] = {0x01};
//uint8_t WFDL_data[] = {0xDF, 0x18,0x00};
uint8_t WTOC_data_1A[] = {0x92, 0x02, 0x00, 0x00, 0x00, 0x00};
uint8_t WTOC_data_2A[] = {0x92, 0x03, 0x00, 0x00, 0x00, 0x00};
uint8_t WPC_dataA[] = {0x96, 0x01, 0x01, 0x07, 0xF8, 0x2A, 0x00, 0x00, 0x98, 0x08, 0x00, 0x00, 0x88, 0x13, 0x00, 0x00};
uint8_t WOM_dataA[] = {0x05, 0x03};

	
//Internal mode commands
 //uint8_t WTOC_data_1[] = {0x92, 0x02, 0x00, 0x00, 0x00, 0x00};
 //uint8_t WTOC_data_2[] = {0x92, 0x07, 0x00, 0x00, 0x00, 0x00};
//uint8_t WTIC_data[] = {0x90, 0x02};
//uint8_t WPRC_data[] = {0x94, 0x02};
//uint8_t WOM_data[] = {0x05, 0x04};
//uint8_t WIPC_data[] = {0x9E, 0x00, 0xFF};
	
void mode_A(void)
{
	//uint8_t RVET_data[] = {0x9D, 0x00, 0x00, 0xD0, 0x07, 0x00, 0x00};
	//uint8_t WTOC_data_1[] = {0x92, 0x00, 0x00, 0x00, 0x00, 0x00};
	//uint8_t WTOC_data_2[] = {0x92, 0x00, 0x00, 0x00, 0x00, 0x00};
	//uint8_t WPC_data[] = {0x96, 0x00, 0x01, 0x04, 0xD0, 0x07, 0x00, 0x00, 0x88, 0x13, 0x00, 0x00, 0xD0, 0x07, 0x00, 0x00};
	//uint8_t WOM_data[] = {0x05, 0x03};
	//I2C_2_dlp_rgb_common(RVET_data, 7);
	//delay_ms(5);
	//I2C_2_dlp_read_common(R_H01_data, 1);
	//delay_ms(6);
	//I2C_2_dlp_rgb_common(WFDTS_data, 2);
	//I2C_2_dlp_rgb_common(WFDL_data, 3);
	I2C_2_dlp_rgb_common(WTOC_data_1A, 6); //e
	delay_ms(20);
	I2C_2_dlp_rgb_common(WTOC_data_2A, 6); //e
	delay_ms(20);
	I2C_2_dlp_rgb_common(WPC_dataA, 16); //e
	//delay_ms(6);
	//I2C_2_dlp_rgb_common(WTIC_data, 2); //i
	//I2C_2_dlp_rgb_common(WPRC_data, 2); //
	delay_ms(20);
	I2C_2_dlp_rgb_common(WOM_dataA, 2); //e
	//I2C_2_dlp_rgb_common(WIPC_data, 3); //i
	//I2C_2_dlp_rgb_common(ROMS_data, 1);

}

uint8_t WTOC_data_1B[] = {0x92, 0x02, 0x00, 0x00, 0x00, 0x00};
uint8_t WTOC_data_2B[] = {0x92, 0x03, 0x00, 0x00, 0x00, 0x00};
//uint8_t WPC_dataB[] = {0x96, 0x01, 0x01, 0x07, 0xF8, 0x2A, 0x00, 0x00, 0x98, 0x08, 0x00, 0x00, 0x88, 0x13, 0x00, 0x00};
uint8_t WPC_dataB[] = {0x96, 0x02, 0x03, 0x04, 0x88, 0x13, 0x00, 0x00, 0x5E, 0x01, 0x00, 0x00, 0xCE, 0x00, 0x00, 0x00};
uint8_t WOM_dataB[] = {0x05, 0x03};
void mode_B(void)
{
	I2C_2_dlp_rgb_common(WTOC_data_1B, 6); //e
	delay_ms(20);
	I2C_2_dlp_rgb_common(WTOC_data_2B, 6); //e
	delay_ms(20);
	I2C_2_dlp_rgb_common(WPC_dataB, 16); //e
	delay_ms(20);
	I2C_2_dlp_rgb_common(WOM_dataB, 2); //e
}
uint8_t WTOC_data_1C[] = {0x92, 0x02, 0x00, 0x00, 0x00, 0x00};
uint8_t WTOC_data_2C[] = {0x92, 0x03, 0x00, 0x00, 0x00, 0x00};
uint8_t WPC_dataC[] = {0x96, 0x00, 0x18, 0x04, 0xC3, 0x01, 0x00, 0x00, 0xC1, 0x00, 0x00, 0x00, 0x32, 0x00, 0x00, 0x00};
uint8_t WOM_dataC[] = {0x05, 0x03};
void mode_C(void)
{
		I2C_2_dlp_rgb_common(WTOC_data_1C, 6); //e
		delay_ms(20);
		I2C_2_dlp_rgb_common(WTOC_data_2C, 6); //e
		delay_ms(20);
		I2C_2_dlp_rgb_common(WPC_dataC, 16); //e
		delay_ms(20);
		I2C_2_dlp_rgb_common(WOM_dataC, 2); //e
}

void triggout_sync_callback(void)
{
	interrupt_count++;
	if (interrupt_count >= 2) { // Toggle PWM state every second interrupt (60Hz input, 30Hz output)
		interrupt_count = 0;
		triggout_state = !triggout_state;
		gpio_set_pin_level(Sens_trig_out_1, triggout_state); // Assume PWM_PIN is defined and configured as an output
	}
}

int main(void)
{
	/* Initializes MCU, drivers and middleware */
	atmel_start_init();
	set_XVS_freq(); //XVS, PA15
	set_XHS_freq(); //XHS, PA23
	pwm_enable(&PWM_0);
	pwm_enable(&PWM_1);
	//i2C, SDA=PA16, SCL=PA17

	//IRQ
	ext_irq_register(DMD_Trig1, triggout_sync_callback); //DMD_Trig1=PC00, DMD_Trig2=PC01

	//USART, TX=PB24, RX=PB25
	configure_usart();

	volatile uint32_t intenset = SERCOM0->USART.INTENSET.reg;
	SERCOM0->USART.INTENSET.reg = 0b00000111;
	

	
	while (1) {
		if (data_received) {
			data_received = false;

			// Run the appropriate mode based on mode_byte
			switch (mode_byte) {
				case 'A':
				mode_A();
				break;
				case 'B':
				mode_B();
				break;
				case 'C':
				mode_C();
				break;
				default:
				// Handle invalid mode byte
				break;
			}

				// Extract I2C data from rx_buffer
				uint8_t I2C_2_data_RGB[2] = { rx_buffer[2], rx_buffer[3] };
				// Extract I2C_2_data_RGB_current values from rx_buffer
				uint8_t I2C_2_data_RGB_current[7];
				for (uint8_t i = 0; i < 7; i++) {
					I2C_2_data_RGB_current[i] = rx_buffer[4 + i];
				}

				I2C_2_dlp_rgb_common(I2C_2_data_RGB, 2);
				I2C_2_dlp_rgb_common(I2C_2_data_RGB_current, 7);



			// Echo received data back for debugging
			struct io_descriptor *io;
			usart_async_get_io_descriptor(&USART_0, &io);
			io_write(io, rx_buffer, 4);
		}

		delay_ms(500);
	}
}