/*
 * PicoPinDef.h
 *
 *  Created on: 2016/1/7
 *      Author: june.liao
 */

#ifndef PICOPINDEF_H_
#define PICOPINDEF_H_

//Port 1
#define PROJ_ON_MCU 		BIT0	// output
#define SEN_ROUT			BIT1	// input
#define SEN_GOUT			BIT2	// input
#define SEN_BOUT			BIT3	// input
#define TRIG_IN_A			BIT4	// input
#define TRIG_IN_B			BIT5	// input
#define MCU_ACK				BIT6	// output
#define MCU_REQ				BIT7	// input

//Port 2
#define DPP_VSYNC_OUT		BIT0	// input
#define FAN1_LOCK			BIT1	// input
#define ITE6801_RST			BIT2	// ourput
#define FAN2_LOCK			BIT3	// input
#define FAN1_PWM			BIT4	// output
#define FAN2_PWM			BIT5	// output
#define IT6801_INTN			BIT6	// input
#define MCU_SW_ONOFF		BIT7	// input

//Port 3
#define MCU_SLAVE_SDA		BIT0	// BI
#define MCU_SLAVE_SCL		BIT1	// BI
#define MCU_P32				BIT2	// input
#define MCU_P33				BIT3	// input
#define P3P3V_PWR_EN		BIT4	// input (reserved)

//Port 4
#define LED_SYS_ON_OFF		BIT0	// output
#define MCU_MASTER_SDA		BIT1	// BI
#define MCU_MASTER_SCL		BIT2	// BI
#define TP260				BIT3	// input
#define TP259				BIT4	// input
#define TP258				BIT5	// input
#define TP257				BIT6	// input
#define TP256				BIT7	// input

//Port 5
#define TP254				BIT0	// input
#define TP255				BIT1	// input
#define MCU_DPP_RST			BIT2	// output
#define SPI_BUS_SEL			BIT3	// output
#define M_HOST_IRQ			BIT4	// input
#define S_HOST_IRQ			BIT5	// input

//Port 6

#define SW_SYS_ON_OFF		BIT0	// input
#define TP250				BIT1	// input
#define TP251				BIT2	// input
#define TP252				BIT3	// input
#define TP253				BIT4	// input
#define CM_LED_INDICATOR1	BIT5	// output (reserved)
#define CM_LED_INDICATOR2	BIT6	// output (reserved)
#define CM_LED_INDICATOR3	BIT7	// output (reserved)

#endif /* PICOPINDEF_H_ */
