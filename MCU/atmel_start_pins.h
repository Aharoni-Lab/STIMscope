/*
 * Code generated from Atmel Start.
 *
 * This file will be overwritten when reconfiguring your Atmel Start project.
 * Please copy examples or other code you want to keep to a separate file
 * to avoid losing it when reconfiguring.
 */
#ifndef ATMEL_START_PINS_H_INCLUDED
#define ATMEL_START_PINS_H_INCLUDED

#include <hal_gpio.h>

// SAMD51 has 14 pin functions

#define GPIO_PIN_FUNCTION_A 0
#define GPIO_PIN_FUNCTION_B 1
#define GPIO_PIN_FUNCTION_C 2
#define GPIO_PIN_FUNCTION_D 3
#define GPIO_PIN_FUNCTION_E 4
#define GPIO_PIN_FUNCTION_F 5
#define GPIO_PIN_FUNCTION_G 6
#define GPIO_PIN_FUNCTION_H 7
#define GPIO_PIN_FUNCTION_I 8
#define GPIO_PIN_FUNCTION_J 9
#define GPIO_PIN_FUNCTION_K 10
#define GPIO_PIN_FUNCTION_L 11
#define GPIO_PIN_FUNCTION_M 12
#define GPIO_PIN_FUNCTION_N 13

#define Sens_trig_out_1 GPIO(GPIO_PORTA, 12)
#define IMX_XVS GPIO(GPIO_PORTA, 15)
#define PA16 GPIO(GPIO_PORTA, 16)
#define PA17 GPIO(GPIO_PORTA, 17)
#define IMX_XHS GPIO(GPIO_PORTA, 23)
#define PB24 GPIO(GPIO_PORTB, 24)
#define PB25 GPIO(GPIO_PORTB, 25)
#define DMD_Trig1 GPIO(GPIO_PORTC, 0)
#define DMD_Trig2 GPIO(GPIO_PORTC, 1)
#define LED GPIO(GPIO_PORTC, 30)

#endif // ATMEL_START_PINS_H_INCLUDED
