/***************************************************************************** 
**
**             TEXAS INSTRUMENTS PROPRIETARY INFORMATION
**
**  (c) Copyright, Texas Instruments Incorporated, 2008
**      All Rights Reserved.
**
**  Property of Texas Instruments Incorporated. Restricted Rights -
**  Use, duplication, or disclosure is subject to restrictions set
**  forth in TI's program license agreement and associated documentation.
******************************************************************************/
/*************************************************************
* THIS PROGRAM IS PROVIDED "AS IS." TI MAKES NO WARRANTIES OR
* REPRESENTATIONS, EITHER EXPRESS, IMPLIED OR STATUTORY,
* INCLUDING ANY IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS
* FOR A PARTICULAR PURPOSE, LACK OF VIRUSES, ACCURACY OR
* COMPLETENESS OF RESPONSES, RESULTS AND LACK OF NEGLIGENCE.
* TI DISCLAIMS ANY WARRANTY OF TITLE, QUIET ENJOYMENT, QUIET
* POSSESSION, AND NON-INFRINGEMENT OF ANY THIRD PARTY
* INTELLECTUAL PROPERTY RIGHTS WITH REGARD TO THE PROGRAM OR
* YOUR USE OF THE PROGRAM.
*
* IN NO EVENT SHALL TI BE LIABLE FOR ANY SPECIAL, INCIDENTAL,
* CONSEQUENTIAL OR INDIRECT DAMAGES, HOWEVER CAUSED, ON ANY
* THEORY OF LIABILITY AND WHETHER OR NOT TI HAS BEEN ADVISED
* OF THE POSSIBILITY OF SUCH DAMAGES, ARISING IN ANY WAY OUT
* OF THIS AGREEMENT, THE PROGRAM, OR YOUR USE OF THE PROGRAM.
* EXCLUDED DAMAGES INCLUDE, BUT ARE NOT LIMITED TO, COST OF
* REMOVAL OR REINSTALLATION, COMPUTER TIME, LABOR COSTS, LOSS
* OF GOODWILL, LOSS OF PROFITS, LOSS OF SAVINGS, OR LOSS OF
* USE OR INTERRUPTION OF BUSINESS.  IN NO EVENT WILL TI'S
* AGGREGATE LIABILITY UNDER THIS AGREEMENT OR ARISING OUT OF
* YOUR USE OF THE PROGRAM EXCEED FIVE HUNDRED DOLLARS
* (U.S.$500).
*
* Unless otherwise stated, the Program is written and copyrighted
* by Texas Instruments is distributed as "freeware."  You may,
* only under TI's copyright in the Program, use and modify the
* Program without any charge or restriction.  You may
* distribute to third parties, provided that you transfer a
* copy of this license to the third party and the third party
* agrees to these terms by its first use of the Program.  In
* jurisdictions in which use is not deemed acceptance of these
* terms, no license is granted and no use is permitted.  You
* must reproduce the copyright notice and any other legend of
* ownership on each copy or partial copy of the Program.
*
* You acknowledge and agree that the Program contains
* copyrighted material, trade secrets and other TI proprietary
* information and is protected by copyright laws,
* international copyright treaties, and trade secret laws, as
* well as other intellectual property laws.  You agree that in
* no event will you alter, remove or destroy any copyright
* notice included in the Program.  TI reserves all rights not
* specifically granted under this license.  Except as
* specifically provided herein, nothing in this agreement
* shall be construed as conferring upon you, by implication,
* estoppel, or otherwise, any license or other right under any
* TI patents, copyrights or trade secrets.
*************************************************************/
#include "common.h"
#include "msp430f5514.h"
#include "i2c_master.h"

// local variables
uint08 i2c_desired_byte_count;
uint16 i2c_baseline_timer;
uint08* i2c_data_ptr;
uint08 i2c_actual_byte_count;
uint08 i2c_complete = 0;
uint08 i2c_return_val = 0;

//uint08 I2C_RxBuffer[27]={0};
//uint08 RXByteCtr = 0;
//BOOL STP_Rx = FALSE;


//unsigned char Rx_Buffer = 0;
//unsigned char Rx_Flag = 0x00;
//unsigned char Tx_Flag = 0x00;

//unsigned char TXData = 0, TXByteCtr = 0;
//unsigned char RXData = 0;
extern BOOL I2C_GET_DATA ;

//int UART_Mode = 0;


// local functions
void i2c_master_init(void);
void i2c_master_set_clockrate(uint32 rate);
void i2c_master_config_timer(void);
uint08 i2c_master_check_nak_timeout(uint08 timeout, uint08 check_nak);

void i2c_master_init(void)
{
  // pin setup should have been done prior to calling this function
  // make sure it is in reset
	UCB1CTL1 = UCSWRST;

  // master mode configuration
	UCB1CTL0 = UCMST + UCMODE_3 + UCSYNC;
	UCB1CTL1 |= UCSSEL_2; //SMCLK
}

void i2c_master_config_timer(void)
{
  // configure Timer B for use with the timeout checking of read or write operations
  TBCTL = TBSSEL_2 + MC_2 + ID_3;
}

void i2c_master_set_clockrate(uint32 scl_rate)
/**
 * Configure the SCL clock rate of the I2C bus
 *
 * @param   scl_rate   - I - desired SCL frequency; i.e. 100,000 or 400,000
 *
 */
{
  // can I assume the system clock is 1MHz?
	UCB1BR0 = 12000000 / scl_rate;
  //UCB0BR0 = 16000000 / scl_rate;
  
  // if system clock is highest (16MHz) and SCL rate is lowest (100KHz)
  //  then this register will never be used
	UCB1BR1 = 0;
}

uint08 i2c_master_polled_write(uint08 device_addr, uint08* write_data, uint08 num_bytes, uint08* bytes_written, uint08 timeout)
/**
 * Writes data to the specified device address
 *
 * @param   device_addr   - I - 7 Bit device Address
 * @param   write_data    - I - Pointer to data buffer to be written to slave
 * @param   num_bytes     - I - Number of bytes to be written
 * @param   bytes_written - O - Actual number of bytes written to slave
 * @param   timeout       - I - Timeout max (in msec) between each transmitted byte
 *                              Timeout=0, permits an infinite timeout (not recommended). Max value = 65
 * @return  0 - Completed successfully 
 *          I2C_NO_ACK - Slave NAck'ed            
 *          I2C_WRITE_TIMEOUT - Slave did not respond before timeout period expired
 *          I2C_INVALID_TIMEOUT - timeout parameter is greater than 65
 *
 */
{
  uint08 status;
  uint08* data_ptr = write_data;
  uint08 i;
  uint16 current_sr;
  uint08 int_disabled = 0;
  
  // check for argument validity
  if ( timeout > 65 )
    return I2C_INVALID_TIMEOUT;
    
  // since this version is polled, we don't want an interrupt firing
  //  and the ISR executing
  current_sr = __get_SR_register();
  if ( current_sr & GIE ) {
    int_disabled = 1;
    __disable_interrupt();
  }
  
  // turn on the timer
  TBCTL |= MC_2;
  i2c_baseline_timer = TBR;
  
  // write the slave address and configure i2c hardware
  UCB1CTL1 = UCSWRST;
  UCB1CTL1 = UCSSEL_2 + UCSWRST;
  UCB1I2CSA = device_addr>>1;
  UCB1CTL1 &= ~UCSWRST;

  // enable module interrupts and start the transfer
  UCB1IE |= UCNACKIE;
  UCB1IE |= UCTXIE;
  //IE2 |= UCB1TXIE;
  UCB1CTL1 |= UCTR + UCTXSTT;
  
  *bytes_written = 0;
  
  // special handling of 1 byte case
  if ( num_bytes == 1 ) {
    while ( (UCB1IFG & UCTXIFG) == 0 ) {
      status = i2c_master_check_nak_timeout(timeout, 1);
      
      if ( status != 0 ) {
        i2c_master_cleanup();
        if ( int_disabled )
          __enable_interrupt();
        return status;
      }
    }
    
    UCB1TXBUF = *data_ptr;
    while ( (UCB1IFG & UCTXIFG) == 0 ) {
      status = i2c_master_check_nak_timeout(timeout, 1);
      
      if ( status != 0 ) {
        i2c_master_cleanup();
        if ( int_disabled )
          __enable_interrupt();
        return status;
      }
    }
    *bytes_written = 1;
  }
  else {
    for ( i=0; i<num_bytes; i++ ) {
      // spin until transmit buffer needs to be reloaded
      while ( (UCB1IFG & UCTXIFG) == 0 ) {
        status = i2c_master_check_nak_timeout(timeout, 1);
        
        if ( status != 0 ) {
          i2c_master_cleanup();
          if ( int_disabled )
            __enable_interrupt();
          return status;
        }
      }
      
      //UCB1IV &=~USCI_I2C_UCTXIFG;
      // load another data byte
      UCB1TXBUF = *data_ptr;
      
      // increment actual byte counter
      *bytes_written = *bytes_written + 1;
      
      // increment the pointer address
      data_ptr++;
      
      // take a new baseline timer value
      i2c_baseline_timer = TBR;
    }
  }
  
  // wait until the last byte has been transferred
  while ( (UCB1IFG & UCTXIFG) == 0 ) {
    status = i2c_master_check_nak_timeout(timeout, 1);
    
    if ( status != 0 ) {
      i2c_master_cleanup();
      if ( int_disabled )
        __enable_interrupt();
      return status;
    }
  }
  // generate a stop
  UCB1CTL1 |= UCTXSTP;
  while (UCB1STAT & UCBBUSY){};

  // turn off the timer to save power
  TBCTL &= ~MC_2;

  UCB1IFG &= ~UCTXIFG;

  // no errors, all bytes transmitted
  UCB1IE &= ~UCNACKIE;
  UCB1IE &= ~UCTXIE;
  if ( int_disabled )
    __enable_interrupt();
  return 0;
}

uint08 i2c_master_polled_read(uint08 device_addr, uint08* read_data, uint08 num_bytes, uint08* bytes_read, uint08 timeout)
/**
 * Reads data from the specified device address
 *
 * @param   device_addr   - I - 7 Bit device Address
 * @param   read_data     - I - Pointer to buffer to hold received data from slave
 * @param   num_bytes     - I - Number of bytes to be read
 * @param   bytes_read    - O - Actual number of bytes read from slave
 * @param   timeout       - I - Timeout max (in msec) between each received byte
 *                              Timeout=0, permits an infinite timeout (not recommended). Max value = 65
 * @return  0 - Completed successfully 
 *          I2C_NO_ACK - Slave NAck'ed            
 *          I2C_TIMEOUT - Slave did not respond before timeout period expired
 *          I2C_INVALID_TIMEOUT - timeout parameter is greater than 65
 *
 */
{
  uint08 status;
  uint08* data_ptr = read_data;
  uint08 i;
  uint16 current_sr;
  uint08 int_disabled = 0;
  
  // check for argument validity
  if ( timeout > 65 )
    return I2C_INVALID_TIMEOUT;
    
  // since this version is polled, we don't want an interrupt firing
  //  and the ISR taking over
  current_sr = __get_SR_register();
  if ( current_sr & GIE ) {
    int_disabled = 1;
    __disable_interrupt();
  }
    
  // turn on the timer
  TBCTL |= MC_2;
  i2c_baseline_timer = TBR;

  // write the slave address and kick off the transfer

  UCB1CTL1 = UCSWRST;
  UCB1CTL1 = UCSSEL_2 + UCSWRST;
  UCB1I2CSA = device_addr>>1;
  UCB1CTL1 &= ~UCSWRST;
  UCB1IE |= UCNACKIE;
  //IE2 |= UCB0RXIE;

  UCB1CTL1 |= UCTXSTT;
    
  if ( num_bytes == 1 ) {
    // spin until start condition cleared
    while ( UCB1CTL1 & UCTXSTT ) {
      status = i2c_master_check_nak_timeout(timeout, 1);
      
      if ( status != 0 ) {
        i2c_master_cleanup();
        if ( int_disabled )
          __enable_interrupt();
        return status;
      }
    }
    UCB1CTL1 |= UCTXSTP;
    while ( (UCB1IFG & UCRXIFG) == 0 ) {
      status = i2c_master_check_nak_timeout(timeout, 1);
      
      if ( status != 0 ) {
        i2c_master_cleanup();
        if ( int_disabled )
          __enable_interrupt();
        return status;
      }
    }
    *read_data = UCB1RXBUF;
    *bytes_read = 1;
  } else {
    *bytes_read = 0;
    
    for ( i=0; i<num_bytes; i++ ) {
      // spin until data is available
      while ( (UCB1IFG & UCRXIFG) == 0 ) {
        status = i2c_master_check_nak_timeout(timeout, 1);
        
        if ( status != 0 ) {
          i2c_master_cleanup();
          if ( int_disabled )
            __enable_interrupt();
          return status;
        }
      }
      
      // increment actual byte counter
      *bytes_read = *bytes_read + 1;
      
      // copy received byte
      *data_ptr = UCB1RXBUF;
      
      // increment the pointer address
      data_ptr++;
      
      // take a new baseline timer value
      i2c_baseline_timer = TBR;
  
      // send stop after the next byte
      if ( i == (num_bytes-2) )
    	  UCB1CTL1 |= UCTXSTP;
    }
  }    
  
  // turn off the timer to save power
  while (UCB1STAT & UCBBUSY){};
  TBCTL &= ~MC_2;
  UCB1IE &= ~UCNACKIE;
  UCB1IE &= ~UCRXIE;
  //UCB0I2CIE &= ~UCNACKIE;
  //IE2 &= ~UCB0RXIE;
  
  // no errors, all bytes received
  if ( int_disabled )
    __enable_interrupt();
  return 0;
}

uint08 i2c_master_check_nak_timeout(uint08 timeout, uint08 check_nak)
/**
 * 
 * @param   timeout   - I - Timeout max (in msec)
 *                          Timeout=0, permits an infinite timeout (not recommended). Max value = 65
 * @param   check_nak - I - Interrupt driven routines do not need to check for no acknowledge from
 *                          slave devices. 1=check for nak, 0=do not check for nak
 *
 * @return  0 - Completed successfully 
 *          I2C_NO_ACK - Slave NAck'ed            
 *          I2C_TIMEOUT - Slave did not respond before timeout period expired
 *
 */
{
  uint16 current_timer_val;
  uint16 timer_diff;
  
  // check for NAK
  if ( check_nak ) {
    if ( UCB1IFG & UCNACKIFG ) {
      // send a stop and clear the interrupt flag
    	UCB1CTL1 |= UCTXSTP;
    	 while (UCB1STAT & UCBBUSY){};
    	//UCB1IV &= ~USCI_I2C_UCNACKIFG;
    	 UCB1IFG &= ~UCNACKIFG;
    	TBCTL &= ~MC_2;
    	UCB1CTL1 = UCSWRST;
    	return I2C_NO_ACK;
    }
  }

  // do we need to check for timeout?
  if ( timeout != 0 ) {
    current_timer_val = TBR;

    // check for TBR rollover since we baselined
    if ( current_timer_val > i2c_baseline_timer )
      timer_diff = current_timer_val - i2c_baseline_timer;
    else
      timer_diff = (65535 - i2c_baseline_timer) + current_timer_val;
    
    // a count of 1000 is equal to 1ms, approximate with shift by 10
    // have we reached the timeout for this byte?
    //if ( (timer_diff>>10) > timeout ) {
    if ( (timer_diff >>10) >= (timeout) ) {
      // generate a stop and return the error flag
      TBCTL &= ~MC_2;
      UCB1CTL1 |= UCTXSTP;
      while (UCB1STAT & UCBBUSY){};
      return I2C_TIMEOUT;
    }
  }
  
  // nak or timeout did not occur
  return 0;
}

void i2c_master_cleanup(void)
/**
 * Cleanup after a failed transaction. Turn off the timer, disable interrupts,
 * and put the I2C hardware into reset.
 *
 */
{
  // turn off the timer to save power
  TBCTL &= ~MC_2;
  
  // clear interrupt flags
  UCB1IFG &= ~UCTXIFG;
  //IFG2 &= ~UCB0TXIFG;
  
  // no errors, all bytes transmitted
  UCB1IE &= ~UCNACKIE;
  UCB1IE &= ~UCTXIE;
  //UCB0I2CIE &= ~UCNACKIE;
  //IE2 &= ~UCB0TXIE;

  // put i2c hardware into reset  
  UCB1CTL1 = UCSWRST;
}



void i2c_master_setup(uint32 rate)
{
	i2c_master_init();
	i2c_master_set_clockrate(rate);
	i2c_master_config_timer();

}





