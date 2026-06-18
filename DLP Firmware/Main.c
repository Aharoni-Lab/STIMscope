/*
 * Main.c
 *
 *  Created on: 2015/12/9
 *      Author: june.liao
 */
//#include <msp430.h>
#include "common.h"
#include "main.h"
#include "msp430f5514.h"
#include "picopindef.h"
#include "i2c_master.h"
#include "DPP343x.h"
#include "ITE6801.h"

unsigned char *PRxData;                     // Pointer to RX data
unsigned char RXByteCtr;
unsigned char RxBuffer[64];

unsigned char *PTxData;                     // Pointer to RX data
unsigned char TXByteCtr;
unsigned char TxBuffer[64];
DDP343XCMD* cmdinfo = 0;

BOOL TURN_ON = FALSE;
uint08 FSMStatus = 0;
uint08 PBStatus = CheckBtnStatus;
uint08 CypressGPIO7Status = CheckBtnStatus;
BOOL SplashDisplay = FALSE;
BOOL pollexinputflag = FALSE;
uint08 Source_sel = 0;

BOOL EXTERNALSPI_OFF = FALSE;
BOOL FLASH_MODE_MASTER = FALSE;
BOOL FLASH_MODE_SLAVE = FALSE;


void Extern_Source_Enable(BOOL flag)
{
	if (flag)
		P2OUT |= ITE6801_RST;
	else
		P2OUT &= ~ITE6801_RST;

}
BOOL Power_on_sequence(void)
{
	BOOL WaitDPP3438Ready;
	uint08 errcnt;
	BOOL errsta;

	P1OUT |= PROJ_ON_MCU;

	WaitDPP3438Ready = TRUE;
	errcnt =0;
	errsta = FALSE;
	while (WaitDPP3438Ready)
	{
		if ((P5IN & M_HOST_IRQ)==0)
		{
			errsta = FALSE;
			WaitDPP3438Ready = FALSE;
		}
		else
		{
			errcnt++;
			__delay_cycles(240000);
			if (errcnt>=100)//maybe 1s
			{
				errsta = TRUE;
				WaitDPP3438Ready = FALSE;
			}
		}
	}
	if (errsta)
		return FALSE;
	errcnt =0;
	WaitDPP3438Ready = TRUE;
	while (WaitDPP3438Ready)
	{
		if ((P5IN & S_HOST_IRQ)==0)
		{
			errsta = FALSE;
			WaitDPP3438Ready = FALSE;
		}
		else
		{
			errcnt++;
			__delay_cycles(240000);
			if (errcnt>=100)
			{
				errsta = TRUE;
				WaitDPP3438Ready = FALSE;
			}
		}
		//if ((P5IN & S_HOST_IRQ)==0)
			//WaitDPP3438Ready = FALSE;
	}
	if (errsta)
		return FALSE;

	P4OUT &=~(LED_SYS_ON_OFF);

	return TRUE;
}

void Power_Off(void)
{
	P1OUT &= ~PROJ_ON_MCU;
	__delay_cycles(1600000);
	P4OUT |= LED_SYS_ON_OFF; // 1.8v off
	__delay_cycles(1000);
}


void PortInit(void)
{
	P1DIR |= (PROJ_ON_MCU | MCU_ACK);
	P2DIR |= (ITE6801_RST|FAN1_PWM|FAN2_PWM |MCU_SW_ONOFF );
	P4DIR |= LED_SYS_ON_OFF;
	P5DIR = 0;

	P1OUT &= ~(PROJ_ON_MCU);
	P1OUT &= ~(MCU_ACK);
	P2OUT &= ~(ITE6801_RST);
	P2OUT |= MCU_SW_ONOFF;
	P4OUT |=(LED_SYS_ON_OFF|MCU_MASTER_SDA|MCU_MASTER_SCL);

	P3SEL |= (MCU_SLAVE_SDA | MCU_SLAVE_SCL);
	PMAPPWD = 0x02D52;                        // Enable Write-access to modify port mapping registers
	P4MAP2 = PM_UCB1SCL;
	P4MAP1 = PM_UCB1SDA;
	PMAPPWD = 0;                              // Disable Write-Access to modify port mapping registers

	P4SEL |= (MCU_MASTER_SDA|MCU_MASTER_SCL);


	P2SEL |= (FAN1_PWM|FAN2_PWM);                       // P2.4 and P2.5 options select
	P2DIR &= ~(MCU_SW_ONOFF );

}

void TimerA0_setup(void)
{
	// SMCLK = 16M / TimerA0 = SMCLK/8 = 2M
	TA0CCR0 = 50000;
	TA0CTL = TASSEL_2 | MC_1| ID_3 | TACLR ;         // SMCLK, upmode, clear TAR
	TA0CCTL0 = CCIE;                          // CCR0 interrupt enabled
}

void TimerA2_Setup(void)
{
	// SMCLK = 16M / TimerA2 = SMCLK = 16M
	  TA2CCR0 = 160;                          // PWM Period/2 //pwm freq = 100khz
	  TA2CCTL1 = OUTMOD_6;                      // CCR1 toggle/set
	  TA2CCR1 = 140;                             // CCR1 PWM duty cycle
	  TA2CCTL2 = OUTMOD_6;                      // CCR2 toggle/set
	  TA2CCR2 = 140;                             // CCR2 PWM duty cycle
	  TA2CTL = TASSEL_2 | MC_3 | TACLR;         // SMCLK, up-down mode, clear TAR
}

void TimerA2_FanOff(void)
{
	TA2CCR1 = 0;
	TA2CCR2 = 0;
	TA2CTL |= TACLR;

	//TA2CCR0 = 160;                          // PWM Period/2 //pwm freq = 100khz
	//TA2CCTL1 = OUTMOD_6;                      // CCR1 toggle/set
	//TA2CCR1 = 0;                             // CCR1 PWM duty cycle
	//TA2CCTL2 = OUTMOD_6;                      // CCR2 toggle/set
	//TA2CCR2 = 0;                             // CCR2 PWM duty cycle
	//TA2CTL = TASSEL_2 | MC_3 | TACLR;         // SMCLK, up-down mode, clear TAR
}

void TimerA2_FanOn(void)
{
	TA2CCR1 = 140;
	TA2CCR2 = 140;
	TA2CTL |= TACLR;
}

void SlaveI2CSetup(void)
{
	UCB0CTL1 |= UCSWRST;                      // Enable SW reset
	UCB0CTL0 = UCMODE_3 | UCSYNC;             // I2C Slave, synchronous mode
	UCB0I2COA = DPP343X_DEV_ADDR>>1;                         // Own Address is 048h
	UCB0CTL1 &= ~UCSWRST;                     // Clear SW reset, resume operation
	UCB0IE |= UCRXIE|UCTXIE | UCSTPIE | UCSTTIE;  //UCTXIE                       // Enable RX interrupt
}


void main(void)
{
	uint08 key_status;
	uint08 PreBtnStatus = TRUE;
	uint08 CurBtnStatus = TRUE;
	uint08 BtnCount = 0;
	uint08 ItePreStatus = VSTATE_Off;
	uint08 IteCurStatus = VSTATE_Off;
	uint08 SwOnOffPreStatus = MCU_SW_ONOFF;
	uint08 SwOnOffCurStatus = MCU_SW_ONOFF;
	BOOL H_S_IRQ_Sta;


	WDTCTL = WDTPW|WDTHOLD;                   // Stop WDT
	P1DIR = 0;
	P2DIR = 0;
	P2DIR = 0;
	P3DIR = 0;
	P4DIR = 0;
	P5DIR = 0;
	P6DIR = 0;

	P1OUT = 0;
	P2OUT = 0;
	P3OUT = 0;
	P4OUT = 0;
	P5OUT = 0;
	P6OUT = 0;

	PortInit();

	UCSCTL3 |= SELREF_2;                      // Set DCO FLL reference = REFO
	UCSCTL4 |= SELA_2;                        // Set ACLK = REFO
	UCSCTL0 = 0x0000;                         // Set lowest possible DCOx, MODx
	// Loop until XT1,XT2 & DCO stabilizes - In this case only DCO has to stabilize
	do
	{
		UCSCTL7 &= ~(XT2OFFG | XT1LFOFFG | DCOFFG);
		// Clear XT2,XT1,DCO fault flags
		SFRIFG1 &= ~OFIFG;                      // Clear fault flags
	}
	while (SFRIFG1&OFIFG);                   // Test oscillator fault flag

	__bis_SR_register(SCG0);                  // Disable the FLL control loop
	UCSCTL1 = DCORSEL_5;                      // Select DCO range 16MHz operation
	UCSCTL2 |= 365;                           // Set DCO Multiplier for 16MHz//max value = 512                                      // (N + 1) * FLLRef = Fdco
	                                          // (487 + 1) * 32768 = 16MHz
	__bic_SR_register(SCG0);                // Enable the FLL control loop

	// Worst-case settling time for the DCO when the DCO range bits have been
	// changed is n x 32 x 32 x f_MCLK / f_FLL_reference. See UCS chapter in 5xx
	// UG for optimization.
	// 32 x 32 x 16 MHz / 32,768 Hz = 500000 = MCLK cycles for DCO to settle
	__delay_cycles(500000);

	i2c_master_setup(100000);
	TimerA0_setup();
	TimerA2_Setup();
	SlaveI2CSetup();
	 __bis_SR_register(GIE);

	 Extern_Source_Enable(TRUE);
	 ITE6801_Init();

	while(1)
	{
		CurBtnStatus = (P6IN & SW_SYS_ON_OFF)|0xFE;
		if (CurBtnStatus != 0xFF)
		{
			if (PreBtnStatus == CurBtnStatus)
			{
				if (BtnCount == 8)
				{
					BtnCount++;
					key_status = ~CurBtnStatus;
					if (key_status == SW_SYS_ON_OFF)
					{
						if (!TURN_ON)
						{
							PBStatus = LightEngineON;
						}
						else
						{
							PBStatus = LightEngineOFF;
						}
					}
					else
					{
						PBStatus = CheckBtnStatus;
					}
				}
				else
				{
					if (BtnCount < 0xFF)
						BtnCount++;
					__delay_cycles(240000);
				}
			}
			else
			{
				BtnCount=0;
				PreBtnStatus = CurBtnStatus;
				__delay_cycles(240000);
			}
		}
		else
		{
			BtnCount = 0;
			PreBtnStatus = CurBtnStatus;
			__delay_cycles(240000);
		}

		SwOnOffCurStatus = (P2IN & MCU_SW_ONOFF);
		if (!TURN_ON)
		{
			if ((PBStatus == LightEngineON) && (SwOnOffCurStatus == MCU_SW_ONOFF))
			{
				PBStatus = CheckBtnStatus;
				FSMStatus = LightEngineON;
			}
			else
				PBStatus = CheckBtnStatus;
		}
		else
		{
			if ((PBStatus == LightEngineOFF ) && (SwOnOffCurStatus == MCU_SW_ONOFF))
			{
				PBStatus = CheckBtnStatus;
				FSMStatus = LightEngineOFF;
			}
			else
				PBStatus = CheckBtnStatus;
		}

		if (SwOnOffCurStatus != SwOnOffPreStatus )
		{
			if (SwOnOffCurStatus == MCU_SW_ONOFF)
			{
				if (!TURN_ON)
				{
					//TURN_ON = TRUE;
					FSMStatus = LightEngineON;
					CypressGPIO7Status = LightEngineON;
				}
			}
			else
			{
				if (TURN_ON)
				{
					//TURN_ON = FALSE;
					FSMStatus = LightEngineOFF;
					CypressGPIO7Status = LightEngineOFF;
				}
			}
			SwOnOffPreStatus = SwOnOffCurStatus;
		}
		/****************************** for EVM Test *********************************************/
		if ((!TURN_ON) && (EXTERNALSPI_OFF == TRUE))
		{

			P5DIR |= (MCU_DPP_RST | SPI_BUS_SEL) ;//out
			if (((P3IN & MCU_P32) == MCU_P32 ) && ((P3IN & MCU_P33) == MCU_P33))//flash master
			{
				if (FLASH_MODE_MASTER == FALSE)
				{
					P5OUT &= ~(MCU_DPP_RST ) ;
					P5OUT |= SPI_BUS_SEL;
					FLASH_MODE_MASTER = TRUE;
				}
			}
			else if (((P3IN & MCU_P32) == 0 ) && ((P3IN & MCU_P33) == 0))//flash slave
			{

				P5DIR |= (MCU_DPP_RST | SPI_BUS_SEL);
				if (FLASH_MODE_SLAVE == FALSE)
				{
					P5OUT &= ~(MCU_DPP_RST ) ;
					P5OUT &= ~(SPI_BUS_SEL ) ;
					FLASH_MODE_SLAVE = TRUE;
				}
			}
		}
		if (((P3IN & MCU_P32) == 0) && ((P3IN & MCU_P33) == MCU_P33))
		{
			TimerA2_FanOn();
			if (!TURN_ON)
			{
				FSMStatus = LightEngineON;
				TURN_ON = TRUE;

			}
		}
		if (((P3IN & MCU_P32) == MCU_P32) && ((P3IN & MCU_P33) == 0))
		{
			EXTERNALSPI_OFF = TRUE;
			TimerA2_FanOff();
			if (TURN_ON)
			{
				FSMStatus = LightEngineOFF;
				TURN_ON = FALSE;
			}
		}
		/*****************************************************************************************/

		if (TURN_ON)
		{
			if ((P1IN & MCU_REQ) == MCU_REQ)
			{
				P1OUT |= MCU_ACK;
				continue;
			}
			else
			{
				P1OUT &= ~(MCU_ACK);
			}

			if (pollexinputflag)
			{
				ITE6801_polling_input();
				IteCurStatus = GetITE6801CurStatus();
				if (ItePreStatus != IteCurStatus)
				{
					if (IteCurStatus == VSTATE_VideoOn)
						FSMStatus = ExternalSourceDisplay;
					else
					{
						if (!SplashDisplay)
							FSMStatus = SplashOn;
					}

					ItePreStatus = IteCurStatus;
				}
			}
		}

		switch (FSMStatus)
		{
			case CheckBtnStatus:
				FSMStatus = CheckBtnStatus;
			break;
			case LightEngineON:
				TURN_ON = TRUE;
				EXTERNALSPI_OFF= FALSE;
				FLASH_MODE_MASTER = FALSE;
				FLASH_MODE_SLAVE = FALSE;
				P5DIR &= ~(MCU_DPP_RST | SPI_BUS_SEL) ;
				H_S_IRQ_Sta = Power_on_sequence();
				if (H_S_IRQ_Sta)
				{
					pollexinputflag = TRUE;
					SplashDisplay = TRUE;
					if (IteCurStatus == VSTATE_VideoOn)
						FSMStatus = ExternalSourceDisplay;
					else
						FSMStatus = CheckBtnStatus;
				}
				else
				{
					TURN_ON = FALSE;
					Power_Off();
					pollexinputflag = FALSE;
					FSMStatus = CheckBtnStatus;

				}
				//FSMStatus = SplashOn;
			break;
			case LightEngineOFF:
				TURN_ON = FALSE;
				Power_Off();
				pollexinputflag = FALSE;
				FSMStatus = CheckBtnStatus;
			break;
			case ExternalSourceDisplay:
				SplashDisplay = FALSE;
				Source_sel = INPUT_EXTERNAL_HDMI;
				dpp343x_source_input_select(Source_sel);
				FSMStatus = CheckBtnStatus;
			break;
			case SplashOn:
				SplashDisplay = TRUE;
				Source_sel = INPUT_SPLASH;
				dpp343x_source_input_select(Source_sel);
				FSMStatus = CheckBtnStatus;
			break;
		}
	}
}


// Timer0 A0 interrupt service routine
#pragma vector=TIMER0_A0_VECTOR
__interrupt void TIMER0_A0_ISR(void)
{

}

//uint08 SScnttemp =0;
//uint08 STcnttemp =0;

#pragma vector = USCI_B0_VECTOR
__interrupt void USCI_B0_ISR(void)
{



  switch(__even_in_range(UCB0IV,12))
  {
  case  0: break;                           // Vector  0: No interrupts
  case  2: break;                           // Vector  2: ALIFG
  case  4: break;                           // Vector  4: NACKIFG
  case  6:                                  // Vector  6: STTIFG
	  RXByteCtr = 0;
	  TXByteCtr = 0;
	  //SScnttemp++;
	  UCB0IFG &= ~UCSTTIFG;
	  break;
  case  8: // Vector  8: STPIFG
	  //STcnttemp++;
	  if (RXByteCtr !=0)
	  {
		  if (cmdinfo != NULL)
		  {
			  if (cmdinfo->type == 0)//Write
			  {
				  if (cmdinfo->id == W_TEST_PATTERN_SELECT)
				  {
					  cmdinfo->wxlen = RXByteCtr;
				  }
				  else if (cmdinfo->id == W_EXT_PAD_ADDRESS)
				  {
					  if(RxBuffer[5]== 0x01)//read
						  SetReadPADDataLen(RxBuffer[4]);
					  else
						  SetWritePADDataLen(RxBuffer[4]);
				  }
				  PRxData = (unsigned char *)RxBuffer;
				  write_dpp343x_i2c(DPP343X_DEV_ADDR, *PRxData, (PRxData+1),cmdinfo->wxlen );
			  }
		  }
	  }
	  UCB0IFG &= ~UCSTPIFG;
	  break;
  case 10:// Vector 10: RXIFG

	  RxBuffer[RXByteCtr]=UCB0RXBUF;
	  if (RXByteCtr == 0)
	  {
		  cmdinfo = (DDP343XCMD*)(GetCmdInfo(&(RxBuffer[0])));
	  }

	  RXByteCtr++;
	  if (cmdinfo != NULL)
	 {
		  if (cmdinfo->type == 1 && cmdinfo->wxlen == RXByteCtr)//Write
		  {
			  PTxData = (unsigned char *)TxBuffer;
			  Read_dpp343x_i2c(DPP343X_DEV_ADDR,&RxBuffer[0] ,cmdinfo->wxlen,PTxData ,cmdinfo->rxlen);
		  }
	 }

	  break;
  case 12:
	//if (TotalTxLen!=0)
	{
		UCB0TXBUF = TxBuffer[TXByteCtr];
		TXByteCtr++;
	}
	//else
		//UCB0TXBUF = 0xFF;
	break;                           // Vector 12: TXIFG
  default:
	break;
  }
}



