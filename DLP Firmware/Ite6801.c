/*
 * Ite6801.c
 *
 *  Created on: 2014/12/3
 *      Author: june.liao
 */
#include "common.h"
#include "ITE6801.h"
#include "i2c_master.h"
#include "msp430x22x2.h"

uint08 Cur_VSTATE =0;
uint08 m_bInputVideoMode =0;
uint08 m_bOutputVideoMode = 0;
uint08 RGBQuantizationRange =0;
uint08 VIC =0;
uint08 m_NewAVIInfoFrameF =0;
uint08 m_VidOutDataTrgger =0;
uint08 m_VidOutSyncMode =0;
uint08 bSynWaitcnt = 0;
BOOL bSynWaitEn = FALSE;

uint08 rxmsgdata[2];
uint08 txmsgdata[2];
uint08 CBusIntEvent;
uint08 CBusWaitNo;
uint08 RCPResult;
uint08 RCPCheckResponse;
uint08 wakeupfailcnt;
uint08 CBusSeqNo;
uint08 rxscrpad[16];
uint08 m_ucEccCount_P0;
uint08 m_bUpHDMIMode;
uint08 m_ucCurrentHDMIPort;
uint08 ucPortAMPOverWrite[2];
uint08 HDMIIntEvent;
uint08 HDMIWaitNo[2];
uint08 m_RxHDCPState;
uint16 Pico_HActive;
uint16 Pico_VActive;
static void IT6801SwitchVideoState(uint08 eNewVState);


static unsigned char bCSCMtx_YUV2RGB_ITU709_16_235[21] =
{
	0x00,		0x00,		0x00,
	0x00,0x08,	0x55,0x3C,	0x88,0x3E,
	0x00,0x08,	0x51,0x0C,	0x00,0x00,
	0x00,0x08,	0x00,0x00,	0x84,0x0E
} ;

static unsigned char bCSCMtx_YUV2RGB_ITU709_0_255[21] =
{
	0x04,		0x00,		0xA7,
	0x4F,0x09,	0xBA,0x3B,	0x4B,0x3E,
	0x4F,0x09,	0x57,0x0E,	0x02,0x00,
	0x4F,0x09,	0xFE,0x3F,	0xE8,0x10
} ;

static unsigned char bCSCMtx_YUV2RGB_ITU601_16_235[21] =
{
	0x00,		0x00,		0x00,
	0x00,0x08,	0x6B,0x3A,	0x50,0x3D,
	0x00,0x08,	0xF5,0x0A,	0x02,0x00,
	0x00,0x08,	0xFD,0x3F,	0xDA,0x0D
} ;

static unsigned char bCSCMtx_YUV2RGB_ITU601_0_255[21] =
{
	0x04,		0x00,		0xA7,
	0x4F,0x09,	0x81,0x39,	0xDD,0x3C,
	0x4F,0x09,	0xC4,0x0C,	0x01,0x00,
	0x4F,0x09,	0xFD,0x3F,	0x1F,0x10
} ;

/*static unsigned char bCSCMtx_RGB_16_235_RGB_0_255[21] =
{
	0xED,		0xED,		0x00,
	0x50,0x09,	0x00,0x00,	0x00,0x00,
	0x00,0x00,	0x50,0x09,	0x00,0x00,
	0x00,0x00,	0x00,0x00,	0x50,0x09,
} ;

static unsigned char bCSCMtx_RGB_0_255_RGB_16_235[21] =
{
	0x10,		0x10,		0x00,
	0xe0,0x06,	0x00,0x00,	0x00,0x00,
	0x00,0x00,	0xe0,0x06,	0x00,0x00,
	0x00,0x00,	0x00,0x00,	0xe0,0x06,

} ;*/

#ifdef _SUPPORT_RCP_
unsigned char  SuppRCPCode[128]=
{
		1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, // 0
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // 1
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, // 2
        1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, // 3
        0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, // 4
        1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // 5
        1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, // 6
        0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0};// 7
#endif

#ifdef _SUPPORT_RAP_
//                      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, A, B, C, D, E, F
unsigned char  SuppRAPCode[32] =
{
		1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // 0
        1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};// 1
#endif





static uint08 Ite6801RegSet(uint08 slaveaddr,uint08  offset, uint08  mask, uint08  ucdata );

uint08 Write_Ite6801_i2c(uint08 slavaddr, uint08 offset, uint08* data);
uint08 Read_Ite6801_i2c(uint08 slaveaddr, uint08 offset, uint08* data);

uint08 IT6801_Identify_Chip(void)
{
	uint08 acIT6801A0Version[4]={0x54,0x49,0x02,0x68};
	uint08 readdata;
	uint08 status;
	uint08 i;

	for (i=0;i<4;i++)
	{
		status = Read_Ite6801_i2c(ITE_HDMI_I2C_ADDR,i,&readdata);
		if (status == FALSE)
			break;
		if (readdata != acIT6801A0Version[i])
			break;
	}

	if (i<4)
		return FALSE;
	else
		return TRUE;
}

static void hdmi_table_init(void)
{
	Ite6801RegSet(ITE_HDMI_I2C_ADDR,REG_RX_0F,0x03,BANK0);//change Bank 0
	Ite6801RegSet(ITE_HDMI_I2C_ADDR,REG_RX_10,0xFF,0x08);	//[3]1: Register reset
	Ite6801RegSet(ITE_HDMI_I2C_ADDR,REG_RX_10,0xFF,0x17);
	
	Ite6801RegSet(ITE_HDMI_I2C_ADDR,REG_RX_11,0xFF,0x1F);	//Port 0¡G[4]EQ Reset [3]CLKD5 Reset [2]CDR Reset [1]HDCP Reset [0]All logic Reset
	Ite6801RegSet(ITE_HDMI_I2C_ADDR,REG_RX_18,0xFF,0x1F);	//Port 1¡G[4]EQ Reset [3]CLKD5 Reset [2]CDR Reset [1]HDCP Reset [0]All logic Reset

	Ite6801RegSet(ITE_HDMI_I2C_ADDR,REG_RX_12,0xFF,0xF8);	//Port 0¡G[7:3] MHL Logic reset

	Ite6801RegSet(ITE_HDMI_I2C_ADDR,REG_RX_10,0xFF,0x10);	//[4]Auto Video Reset [2]Int Reset [1]Audio Reset [0]Video Reset

	Ite6801RegSet(ITE_HDMI_I2C_ADDR,REG_RX_11,0xFF,0xA0);	//Port 0¡G[7] Enable Auto Reset when Clock is not stable [5]Enable Auto Reset
	Ite6801RegSet(ITE_HDMI_I2C_ADDR,REG_RX_18,0xFF,0xA0);	//Port 1¡G[7] Enable Auto Reset when Clock is not stable [5]Enable Auto Reset

	Ite6801RegSet(ITE_HDMI_I2C_ADDR,REG_RX_12,0xFF,0x00);	//Port 0¡G[7:3] MHL Logic reset

	Ite6801RegSet(ITE_HDMI_I2C_ADDR,REG_RX_0F,0x03,BANK1);	//change bank 1	//2013-0430 Andrew suggestion
	Ite6801RegSet(ITE_HDMI_I2C_ADDR,REG_RX_B0,0x03,0x01);	// MHL Port Set HPD = 0 at Power On initial state
	//Ite6801RegSet(ITE_HDMI_I2C_ADDR,REG_RX_C0,0x80,0x00); //add 2016-02-25
	
	Ite6801RegSet(ITE_HDMI_I2C_ADDR,REG_RX_0F,0x03,BANK0);	//change bank 0	//2013-0430 Andrew suggestion
	Ite6801RegSet(ITE_HDMI_I2C_ADDR,REG_RX_17,0xC0,0x80);	//Port 0¡G[7:6] = 10 invert Port 0 input HCLK , CLKD5I	//2013-0430 Andrew suggestion
	Ite6801RegSet(ITE_HDMI_I2C_ADDR,REG_RX_1E,0xC0,0x00);	//Port 1¡G[7:6] = 00 invert Port 1 input TMDS , CLKD5I	//2013-0430 Andrew suggestion

	Ite6801RegSet(ITE_HDMI_I2C_ADDR,REG_RX_16,0x08,0x08);	//Port 0¡G[3]1: Enable CLKD5 auto power down
	Ite6801RegSet(ITE_HDMI_I2C_ADDR,REG_RX_1D,0x08,0x08);	//Port 1¡G[3]1: Enable CLKD5 auto power down


	Ite6801RegSet(ITE_HDMI_I2C_ADDR,REG_RX_2B,0xFF,0x07);	//FixTek3D
	//FIX_ID_042 xxxxx //Disable HDCP 1.1 feature to avoid compilance issue from ilegal HDCP 1.1 source device
	Ite6801RegSet(ITE_HDMI_I2C_ADDR,REG_RX_31,0xFF,0x2C);	//[7:4]Enable repeater function [3:0] SCL hold time count & Update Ri sel
	Ite6801RegSet(ITE_HDMI_I2C_ADDR,REG_RX_34,0xFF,MHL_ADDR+0x01);
	//Ite6801RegSet(ITE_HDMI_I2C_ADDR,0x49,0xFF,0x09);	//[7:4]Enable repeater function [3:0] SCL hold time count & Update Ri sel

	Ite6801RegSet(ITE_HDMI_I2C_ADDR,REG_RX_35,0x1E,0x14);	//[3:2] RCLKDeltaSel , [1] UseIPLock = 0
	Ite6801RegSet(ITE_HDMI_I2C_ADDR,REG_RX_4B,0x1E,0x14);	//[3:2] RCLKDeltaSel , [1] UseIPLock = 0

	Ite6801RegSet(ITE_HDMI_I2C_ADDR,REG_RX_54,0xFF,0x11);	//[1:0]RCLK frequency select
	Ite6801RegSet(ITE_HDMI_I2C_ADDR,REG_RX_6A,0xFF,0x81);			//Decide which kind of packet to be fully recorded on General PKT register
	Ite6801RegSet(ITE_HDMI_I2C_ADDR,REG_RX_74,0xFF,0xA0);	//[7]Enable i2s and SPDIFoutput [5]Disable false DE output
	Ite6801RegSet(ITE_HDMI_I2C_ADDR,REG_RX_50,0x1F,0x12);	//[4]1: Invert output DCLK and DCLK DELAY 2 Step

	Ite6801RegSet(ITE_HDMI_I2C_ADDR,REG_RX_65,0x0C,0x08);	//[3:2]2=12bits Output color depth

	Ite6801RegSet(ITE_HDMI_I2C_ADDR,REG_RX_7A,0x80,0x80);	//[7]1: enable audio B Frame Swap Interupt
	//	{REG_RX_02D,	0x03,	0x03},	//[1:0] 11: Enable HDMI/DVI mode over-write

	Ite6801RegSet(ITE_HDMI_I2C_ADDR,REG_RX_85,0x02,0x02);	//[1]1: gating avmute in video detect module

	//	{REG_RX_051,	0x80,	0x80},	//[7]1: power down color space conversion logic


	Ite6801RegSet(ITE_HDMI_I2C_ADDR,REG_RX_C0,0x03,0x03);	//[0]1:Reg_P0DisableShadow
	Ite6801RegSet(ITE_HDMI_I2C_ADDR,REG_RX_87,0xFF,0x00);	//[7:1] EDID RAM Slave Adr ,[0]1: Enable access EDID block


	Ite6801RegSet(ITE_HDMI_I2C_ADDR,REG_RX_71,0x08,0x00);	//Reg71[3] RegEnPPColMode must clear to 0 for andrew suggestion 2013-0502
	//FIX_ID_030 xxxxx fixed video lost at 640x480 timing
	Ite6801RegSet(ITE_HDMI_I2C_ADDR,REG_RX_37,0xFF,0x88);	//Reg37 Reg_P0_WCLKValidNum must set to 0xA6 for andrew suggestion 2014-0403
	Ite6801RegSet(ITE_HDMI_I2C_ADDR,REG_RX_4D,0xFF,0x88);	//Reg4D Reg_P1_WCLKValidNum must set to 0xA6 for andrew suggestion 2014-0403
	//FIX_ID_030 xxxxx
	Ite6801RegSet(ITE_HDMI_I2C_ADDR,REG_RX_67,0x80,0x00);	//Reg67[7] disable HW CSCSel

	Ite6801RegSet(ITE_HDMI_I2C_ADDR,REG_RX_7A,0x70,0x70);

	Ite6801RegSet(ITE_HDMI_I2C_ADDR,REG_RX_77, 0x20, 0x20);	 // IT6801 Audio i2s sck and mclk is common pin
	Ite6801RegSet(ITE_HDMI_I2C_ADDR,REG_RX_0F, 0x03, BANK1);	//change bank 1
	Ite6801RegSet(ITE_HDMI_I2C_ADDR,REG_RX_C0, 0x80, 0x80);
	Ite6801RegSet(ITE_HDMI_I2C_ADDR,REG_RX_0F, 0x03, BANK0);	//change bank 0

	Ite6801RegSet(ITE_HDMI_I2C_ADDR,REG_RX_7E,0x40,0x40);

	Ite6801RegSet(ITE_HDMI_I2C_ADDR,REG_RX_52,0x20,0x20);				//Reg52[5] = 1 for disable Auto video MUTE
	Ite6801RegSet(ITE_HDMI_I2C_ADDR,REG_RX_53,(B_VDGatting|B_VIOSel|B_TriVDIO|B_TriSYNC),(B_VIOSel|B_TriVDIO|B_TriSYNC));//HdmiSet(REG_RX_053,(B_VDGatting|B_VIOSel|B_TriVDIO|B_TriSYNC),(B_VIOSel|B_TriVDIO|B_TriSYNC));				//Reg53[7][5] = 01    // for disable B_VDIO_GATTING

	Ite6801RegSet(ITE_HDMI_I2C_ADDR,REG_RX_58,0xFF,0x33);			// Reg58 for 4Kx2K Video output Driving Strength

	//	{REG_RX_059,0xFF,0xAA},			// Reg59 for Audio output Driving Strength

	//RS initial valie
	// 2013/06/06 added by jau-chih.tseng@ite.com.tw
	// Dr. Liu said, reg25/reg3D should set as 0x1F for auto EQ start option.
	Ite6801RegSet(ITE_HDMI_I2C_ADDR,REG_RX_25, 0xFF, 0x1F);//DEFAULT_EQVALUE
	Ite6801RegSet(ITE_HDMI_I2C_ADDR,REG_RX_3D, 0xFF, 0x1F);
	//~jau-chih.tseng@ite.com.tw
	Ite6801RegSet(ITE_HDMI_I2C_ADDR,REG_RX_27, 0xFF, 0x1F);	// B ch
	Ite6801RegSet(ITE_HDMI_I2C_ADDR,REG_RX_28, 0xFF, 0x1F);	// G
	Ite6801RegSet(ITE_HDMI_I2C_ADDR,REG_RX_29, 0xFF, 0x1F);	// R
	Ite6801RegSet(ITE_HDMI_I2C_ADDR,REG_RX_3F, 0xFF, 0x1F);
	Ite6801RegSet(ITE_HDMI_I2C_ADDR,REG_RX_40, 0xFF, 0x1F);
	Ite6801RegSet(ITE_HDMI_I2C_ADDR,REG_RX_41, 0xFF, 0x1F);

	Ite6801RegSet(ITE_HDMI_I2C_ADDR,REG_RX_0F,	0x03, BANK1);	//change bank 1	//2013-0515 Andrew suggestion	for Auto EQ
	Ite6801RegSet(ITE_HDMI_I2C_ADDR,REG_RX_BC,	0xFF, 0x06);	//Reg1BC=0x06		//2013-0515 Andrew suggestion	for Auto EQ
	Ite6801RegSet(ITE_HDMI_I2C_ADDR,REG_RX_B5,	0x03, 0x03);
	Ite6801RegSet(ITE_HDMI_I2C_ADDR,REG_RX_B6,	0x07, 0x03);

	Ite6801RegSet(ITE_HDMI_I2C_ADDR,REG_RX_0F,	0x03,BANK1);
	Ite6801RegSet(ITE_HDMI_I2C_ADDR,REG_RX_22,	0xFF, 0x00);
	Ite6801RegSet(ITE_HDMI_I2C_ADDR,REG_RX_3A,	0xFF, 0x00);
	Ite6801RegSet(ITE_HDMI_I2C_ADDR,REG_RX_26,	0xFF, 0x00);
	Ite6801RegSet(ITE_HDMI_I2C_ADDR,REG_RX_3E,	0xFF, 0x00);

	
	Ite6801RegSet(ITE_HDMI_I2C_ADDR,REG_RX_63,0xFF,0x3F);		//for enable interrupt output Pin
	Ite6801RegSet(ITE_HDMI_I2C_ADDR,REG_RX_73, 0x08, 0x00);		// for HDCPIntKey = false

	Ite6801RegSet(ITE_HDMI_I2C_ADDR,REG_RX_60, 0x40, 0x00);		// disable interrupt mask for NoGenPkt_Rcv
}

static void IT6801_Rst(void)
{
	hdmi_table_init();//hdmi_table_init((void*)IT6801_HDMI_INIT_TABLE);//hdmi_table_init();
	Cur_VSTATE = VSTATE_Off;

}

uint08 GetITE6801CurStatus(void)
{
	return Cur_VSTATE;
}

void GetPicoResolution(uint16 *HActive,uint16 *VActive )
{
	*HActive = Pico_HActive;
	*VActive = Pico_VActive;

}


void IT6801VideoOutputEnable(uint08 flag)
{
	unsigned char H_Hbyte,H_Lbyte,V_Hbyte,V_Lbyte; 
	
	Ite6801RegSet(ITE_HDMI_I2C_ADDR,REG_RX_0F,0x03,BANK0);// change to bank0
	if (flag)
	{
		Read_Ite6801_i2c(ITE_HDMI_I2C_ADDR,REG_RX_9F,&H_Hbyte);
		Read_Ite6801_i2c(ITE_HDMI_I2C_ADDR,REG_RX_9E,&H_Lbyte);
		Pico_HActive  =((H_Hbyte&0x3F)<<8) + H_Lbyte;
		Read_Ite6801_i2c(ITE_HDMI_I2C_ADDR,REG_RX_A4,&V_Hbyte);
		Read_Ite6801_i2c(ITE_HDMI_I2C_ADDR,REG_RX_A5,&V_Lbyte);
		Pico_VActive  = ((V_Hbyte&0xF0)<<4) + V_Lbyte;
		if (((Pico_HActive /1940)>1) ||((Pico_VActive/1100)>1)|| ((Pico_HActive/240)==0) ||((Pico_VActive/240)==0))
			IT6801SwitchVideoState(VSTATE_SyncWait);
		else
			Ite6801RegSet(ITE_HDMI_I2C_ADDR,REG_RX_53, (B_TriVDIO|B_TriSYNC), 0x00);//disable B_TriSYNC B_TriVDIO
	}
	else
		Ite6801RegSet(ITE_HDMI_I2C_ADDR,REG_RX_53,(B_TriVDIO|B_TriSYNC),(B_TriVDIO|B_TriSYNC));// enable B_TriSYNC B_TriVDIO
}

void IT6801_AFE_Rst(void)
{
	uint08 Reg51h;
	uint08 PortReg;

	Ite6801RegSet(ITE_HDMI_I2C_ADDR,REG_RX_0F,0x03,BANK1);// change to bank0

	Read_Ite6801_i2c(ITE_HDMI_I2C_ADDR, REG_RX_51, &Reg51h);
	if (Reg51h & 0x01)
		PortReg = REG_RX_18;
	else
		PortReg = REG_RX_11;

	Ite6801RegSet(ITE_HDMI_I2C_ADDR,PortReg,0x01,0x01);
	__delay_cycles(1000);
	Ite6801RegSet(ITE_HDMI_I2C_ADDR,PortReg,0x01,0x00);

}

static unsigned char IsHDMIMode(void)
{

	unsigned char sys_state_P0;
	//unsigned char sys_state_P1;
	unsigned char ucPortSel;

	Read_Ite6801_i2c(ITE_HDMI_I2C_ADDR, REG_RX_0A, &sys_state_P0);
	Read_Ite6801_i2c(ITE_HDMI_I2C_ADDR, REG_RX_51, &ucPortSel);

	if (((sys_state_P0 & B_P0_HDMI_MODE)==B_P0_HDMI_MODE)&&((ucPortSel & B_PORT_SEL)==F_PORT_SEL_0))
		return TRUE;
	else
		return FALSE;
}

static void SetColorSpaceConvert(void)
{
	unsigned char csc ;
	unsigned char filter = 0 ; // filter is for Video CTRL DN_FREE_GO, EN_DITHER, and ENUDFILT
	uint08 i = 0;

    //HDMIRX_VIDEO_PRINTF(("Input mode is YUV444 "));
    if((m_bOutputVideoMode&F_MODE_CLRMOD_MASK) == 0)
    {
    	switch(m_bInputVideoMode&F_MODE_CLRMOD_MASK)
	    {
	    	case F_MODE_YUV444:
	    		csc = B_CSC_YUV2RGB ;
	        break ;
	        case F_MODE_YUV422:
	        	csc = B_CSC_YUV2RGB ;
	        	if((m_bOutputVideoMode & F_MODE_EN_UDFILT) == F_MODE_EN_UDFILT)// RGB24 to YUV422 need up/dn filter.
	        	{
	        		filter |= B_RX_EN_UDFILTER ;
	        	}
	        	if((m_bOutputVideoMode & F_MODE_EN_DITHER) == F_MODE_EN_DITHER)// RGB24 to YUV422 need up/dn filter.
	        	{
	        		filter |= B_RX_EN_UDFILTER|B_RX_DNFREE_GO;//B_RX_EN_UDFILTER | B_RX_DNFREE_GO ;
	        	}
	        	break ;
	        case 0:
	        	csc = B_CSC_BYPASS ;
	    	break ;
	    }
    }

	if(csc == B_CSC_YUV2RGB)
    {
	if((m_bInputVideoMode & F_MODE_ITU709)== F_MODE_ITU709)
        {
            if((m_bOutputVideoMode & F_MODE_16_235)==F_MODE_16_235)
            {
            	for (i=0;i<21;i++)
            		Ite6801RegSet(ITE_HDMI_I2C_ADDR,REG_RX_70,0xFF,bCSCMtx_YUV2RGB_ITU709_16_235[i]);
            }
            else
            {
            	for (i=0;i<21;i++)
            		Ite6801RegSet(ITE_HDMI_I2C_ADDR,REG_RX_70,0xFF,bCSCMtx_YUV2RGB_ITU709_0_255[i]);
            }
        }
        else
        {
            if((m_bOutputVideoMode & F_MODE_16_235)==F_MODE_16_235)
            {
            	for (i=0;i<21;i++)
            		Ite6801RegSet(ITE_HDMI_I2C_ADDR,REG_RX_70,0xFF,bCSCMtx_YUV2RGB_ITU601_16_235[i]);
            }
            else
            {
            	for (i=0;i<21;i++)//
            		Ite6801RegSet(ITE_HDMI_I2C_ADDR,REG_RX_70,0xFF,bCSCMtx_YUV2RGB_ITU601_0_255[i]);
            }
        }

    }

	Ite6801RegSet(ITE_HDMI_I2C_ADDR,REG_RX_0F,0x03,BANK0);//chgbank(0);
	//Ite6801RegSet(ITE_HDMI_I2C_ADDR,0x65,0x03,csc);
	Ite6801RegSet(ITE_HDMI_I2C_ADDR,REG_RX_65,0x03,csc);

    // set output Up/Down Filter, Dither control
	Ite6801RegSet(ITE_HDMI_I2C_ADDR,REG_RX_67,0x07,filter);


}

static void SetDVIVideoOutput(void)
{
	uint08 IN_CSC_CTRL;
	uint08 RxClkXCNT;
	///////////////SetVideoInputFormatWithoutInfoFrame();//////////
	Read_Ite6801_i2c(ITE_HDMI_I2C_ADDR, REG_RX_71, &IN_CSC_CTRL);

	IN_CSC_CTRL |= B_IN_FORCE_COLOR_MODE;
	IN_CSC_CTRL &= ~(M_INPUT_COLOR_MASK);

	IN_CSC_CTRL |= F_MODE_RGB24;
	Ite6801RegSet(ITE_HDMI_I2C_ADDR,REG_RX_71, 0xFF, IN_CSC_CTRL);

	///////////SetColorimetryByMode();??GetColorimetryByMode////////////
	Read_Ite6801_i2c(ITE_HDMI_I2C_ADDR, REG_RX_9A, &RxClkXCNT);
	//m_bInputVideoMode &=(~F_MODE_ITU709);
	if (RxClkXCNT <0x34)
		m_bInputVideoMode |= F_MODE_ITU709;
	else
		m_bInputVideoMode &=(~F_MODE_ITU709);

	/////////SetColorSpaceConvert()//////////////////////
	SetColorSpaceConvert();

	////SetVideoOutputColorFormat() ////RGB24 only ////////
	Ite6801RegSet(ITE_HDMI_I2C_ADDR,REG_RX_65, 0x30, B_OUTPUT_RGB24);
	//hdmirxset(REG_RX_OUT_CSC_CTRL,(M_OUTPUT_COLOR_MASK),B_OUTPUT_RGB24);
	//SetVideoOutputColorFormat(it6802);	//2013-0502
}

static void GetAVIInfoFrame(void)
{
	uint08 temp;

	Ite6801RegSet(ITE_HDMI_I2C_ADDR,REG_RX_0F,0x03,BANK2);//chgbank(2);
	Read_Ite6801_i2c(ITE_HDMI_I2C_ADDR, REG_RX_17, &temp);//it6802->RGBQuantizationRange = ((hdmirxrd(REG_RX_AVI_DB3)&0x0C)>>2);
	RGBQuantizationRange = (temp & 0x0c)>>2;

	Read_Ite6801_i2c(ITE_HDMI_I2C_ADDR, 0x18, &temp);

	VIC= temp & 0x7F; // it6802->VIC = ((hdmirxrd(REG_RX_AVI_DB4)&0x7F));

	//it6802->YCCQuantizationRange = ((hdmirxrd(REG_RX_AVI_DB5)&0xC0)>>6);
	Ite6801RegSet(ITE_HDMI_I2C_ADDR,REG_RX_0F,0x03,BANK0);//chgbank(0);

//FIX_ID_027 xxxxx Support RGB limited / Full range convert
	if(RGBQuantizationRange == 0 )
	{
		if( VIC >=2 )
		{
			// CE Mode
			RGBQuantizationRange = 1 ; // limited range
		}
		else
		{
			// IT mode
			RGBQuantizationRange = 2 ; // Full range
		}
	}
}

static void SetNewInfoVideoOutput(void)
{
	uint08 temp;
	/////////////SetVideoInputFormatWithInfoFrame(it6802);//////////
	Ite6801RegSet(ITE_HDMI_I2C_ADDR,REG_RX_0F,0x03,BANK2);//chgbank(2)
	Read_Ite6801_i2c(ITE_HDMI_I2C_ADDR, REG_RX_15, &temp);
	Ite6801RegSet(ITE_HDMI_I2C_ADDR,REG_RX_0F,0x03,BANK0);//chgbank(0)
	m_bInputVideoMode &=(~0x03);
	temp = (temp>>5)&0x03;
	switch (temp)
	{
		case 2://B_AVI_COLOR_YUV444
			m_bInputVideoMode |=F_MODE_YUV444;
		break;
		case 1://B_AVI_COLOR_YUV422
			m_bInputVideoMode |=F_MODE_YUV422;
		break;
		case 0://B_AVI_COLOR_RGB24
			m_bInputVideoMode |=F_MODE_RGB24;
		break;
		default:
		break;
	}

	Read_Ite6801_i2c(ITE_HDMI_I2C_ADDR, REG_RX_71, &temp);
	temp &= (~0x04);
	Ite6801RegSet(ITE_HDMI_I2C_ADDR,REG_RX_71,0xFF,temp);


	SetColorSpaceConvert();

	Ite6801RegSet(ITE_HDMI_I2C_ADDR,REG_RX_65, 0x30, B_OUTPUT_RGB24);//SetVideoOutputColorFormat(it6802);	//2013-0502

//	get_vid_info();
//	show_vid_info();

}

static void IT6801_VideoOutputModeSet(void)
{
	unsigned char ucReg51;
	unsigned char ucReg65;
	uint08 temp;

	Read_Ite6801_i2c(ITE_HDMI_I2C_ADDR, REG_RX_51, &ucReg51);
	ucReg51 &=0x9B;
	Read_Ite6801_i2c(ITE_HDMI_I2C_ADDR, REG_RX_65, &ucReg65);
	ucReg65 &=0x0F;

	temp =m_bOutputVideoMode & F_MODE_CLRMOD_MASK;
	switch (temp)//m_bOutputVideoMode & 0x03
	{
		case 0://F_MODE_RGB444
			ucReg65 |= F_MODE_RGB24;
		break;
		case 1://F_MODE_YUV422
			ucReg65 |= F_MODE_YUV422;
		break;
		case 2://F_MODE_YUV444
			ucReg65 |= F_MODE_YUV444;
		break;
	}

	switch(m_VidOutDataTrgger)
	{
		case 0://eSDR:
		break;
		case 1://eHalfPCLKDDR:
			ucReg51|=B_HALF_PCLKC;			// 0x40 half PCLK
		break;
		case 2://eHalfBusDDR:
			ucReg51|=B_OUT_DDR;				// 0x20 half bus
		break;
		case 3://eSDR_BTA1004:
			ucReg65|=B_BTA1004Fmt|B_SyncEmb ;	// 0x80 BTA1004 + 0x40 SyncEmb
		break;
		case 4://eDDR_BTA1004:
			ucReg51|=B_HALF_PCLKC;			// 0x40 half PCLK
			ucReg65|=B_BTA1004Fmt|B_SyncEmb ;	// 0x80 BTA1004 + 0x40 SyncEmb
		break;

	}

	switch(m_VidOutSyncMode)
	{
		case 0://eSepSync:
		break;
		case 1://eEmbSync:
			ucReg65|=B_SyncEmb ;	// 0x40 SyncEmb
		break;
		case 2://eCCIR656SepSync:
			ucReg51|=B_SyncEmb;	// 0x04 CCIR656
		break;
		case 3://eCCIR656EmbSync:
			ucReg51|=B_CCIR656;	// 0x04 CCIR656
			ucReg65|=B_SyncEmb ;	// 0x40 SyncEmb
		break;
	}

	Ite6801RegSet(ITE_HDMI_I2C_ADDR,REG_RX_51, 0xFF, ucReg51);
	Ite6801RegSet(ITE_HDMI_I2C_ADDR,REG_RX_65, 0xFF, ucReg65);

}

static void IT6801VideoOutputConfigure(void)
{


	// Configure Output color space convert

//06-27 disable -->	#ifndef DISABLE_HDMI_CSC
//06-27 disable --> 	it6802->m_bOutputVideoMode = HDMIRX_OUTPUT_VID_MODE ;
//06-27 disable -->	#endif
	BOOL bHdmiMode ;
	uint08 GCP_CD;
	uint08 temp;

	bHdmiMode = IsHDMIMode();

	if (!bHdmiMode)
	{
		SetDVIVideoOutput();
	}
	else
	{
		GetAVIInfoFrame();
		SetNewInfoVideoOutput();

	}
	m_NewAVIInfoFrameF =FALSE;

	Read_Ite6801_i2c(ITE_HDMI_I2C_ADDR, REG_RX_99, &temp);
	temp &= 0xF0;
	GCP_CD = temp>>4;
	switch (GCP_CD)
	{
		case 5:
			Ite6801RegSet(ITE_HDMI_I2C_ADDR,REG_RX_65, 0x0C, 0x04);
		break;
		case 6:
			Ite6801RegSet(ITE_HDMI_I2C_ADDR,REG_RX_65, 0x0C, 0x08);
		break;
		default:
			Ite6801RegSet(ITE_HDMI_I2C_ADDR,REG_RX_65, 0x0C, 0x00);
		break;
	}
	IT6801_VideoOutputModeSet();

}


static void IT6801SwitchVideoState(uint08 eNewVState)
{

	if(Cur_VSTATE == eNewVState)
		return;
	Cur_VSTATE = eNewVState;
//	it6802->m_VideoCountingTimer = GetCurrentVirtualTime(); // get current time tick, and the next tick judge in the polling handler.

	switch(Cur_VSTATE)
	{
		case VSTATE_SWReset:
		{
				IT6801VideoOutputEnable(FALSE);
				IT6801_AFE_Rst();
		}
		break;

		case VSTATE_SyncWait:
		{
				// 1. SCDT off interrupt
				// 2. VideoMode_Chg interrupt
				IT6801VideoOutputEnable(FALSE);

				m_NewAVIInfoFrameF=FALSE;//remark //June
				//it6802->m_ucDeskew_P0=0;
				//it6802->m_ucDeskew_P1=0;

		}
		break;

		case VSTATE_SyncChecking:
		{
			bSynWaitEn = TRUE;
			bSynWaitcnt = 10;

				// 1. SCDT on interrupt
				//AssignVideoVirtualTime(VSATE_CONFIRM_SCDT_COUNT);
				//AssignVideoTimerTimeout(VSATE_CONFIRM_SCDT_COUNT);

				//it6802->m_VideoCountingTimer = VSATE_CONFIRM_SCDT_COUNT;////remark //June



		}
		break;
		case VSTATE_VideoOn:
		{
				IT6801VideoOutputConfigure();
				IT6801VideoOutputEnable(TRUE);
				//IT6802SwitchAudioState(it6802,ASTATE_RequestAudio);
				Ite6801RegSet(ITE_HDMI_I2C_ADDR,REG_RX_84, 0xFF, 0x8F);
				//hdmirxwr(0x84, 0x8F);	//2011/06/17 xxxxx, for enable Rx Chip count
				//xxxxx 2013-0812 @@@@@
					//it6802->m_ucSCDTOffCount=0;////remark //June
				//xxxxx 2013-0812
		}
		break;
	}

}

void it6801PortSelect(unsigned char ucPortSel)
{
	uint08 Change_VSTATE;

	Ite6801RegSet(ITE_HDMI_I2C_ADDR,REG_RX_51, B_PORT_SEL, F_PORT_SEL_0);
	Change_VSTATE = VSTATE_SyncWait;

	IT6801SwitchVideoState(Change_VSTATE);

}
void ITE6801_Init(void)
{

	IT6801_Identify_Chip();
	IT6801_Rst();

	// for Disable EDID RAM function !!!
	Ite6801RegSet(ITE_HDMI_I2C_ADDR, REG_RX_C0, 0x03, 0x03);
	//			hdmirxset(REG_RX_087, 0xFF, 0x00);

	it6801PortSelect(0);


}

void it6801HPDCtrl(unsigned char ucport,unsigned char ucEnable)
{
	if(ucport == 0)
	{
		if(ucEnable == 0)
		{
			//Printf("Port 0 HPD 00000 \r\n");
			Ite6801RegSet(ITE_HDMI_I2C_ADDR,REG_RX_0F,0x03,BANK1);//chgbank(1);
			Ite6801RegSet(ITE_HDMI_I2C_ADDR,REG_RX_B0,0x03,0x01);//hdmirxset(REG_RX_1B0, 0x03, 0x01); //clear port 0 HPD=1 for EDID update
			Ite6801RegSet(ITE_HDMI_I2C_ADDR,REG_RX_0F,0x03,BANK0);//chgbank(0);
		}
		else
		{
			//Printf("Port 0 HPD 11111 \r\n");
			Ite6801RegSet(ITE_HDMI_I2C_ADDR,REG_RX_0F,0x03,BANK1);//chgbank(1);
			Ite6801RegSet(ITE_HDMI_I2C_ADDR,REG_RX_B0,0x03,0x03);//hdmirxset(REG_RX_1B0, 0x03, 0x03); //set port 0 HPD=1
			Ite6801RegSet(ITE_HDMI_I2C_ADDR,REG_RX_0F,0x03,BANK0);//chgbank(0);
		}
	}

}

static void hdmirx_INT_HDMIMode_Chg(unsigned char ucport)
{
	unsigned char ucPortSel;
	Read_Ite6801_i2c(ITE_HDMI_I2C_ADDR, REG_RX_51, &ucPortSel);//ucPortSel = hdmirxrd(REG_RX_051)&B_PORT_SEL;
	ucPortSel &= 0x01;

	if(ucPortSel != ucport)
		return;
//FIX_ID_009 xxxxx

	if(IsHDMIMode())
	{
	   /* if(m_VState==VSTATE_VideoOn)
	    {
		    IT6802SwitchAudioState(it6802,ASTATE_RequestAudio);
		}*/
		m_bUpHDMIMode = TRUE ;
		//IT6802_DEBUG_INT_PRINTF(("#### HDMI/DVI Mode : HDMI ####\n"));
	}
	else
	{
		//IT6802SwitchAudioState(it6802,ASTATE_AudioOff);
		m_NewAVIInfoFrameF=FALSE;
	    if(Cur_VSTATE==VSTATE_VideoOn)
	    {
	    	SetDVIVideoOutput();
	    }
		m_bUpHDMIMode = FALSE ;
		//IT6802_DEBUG_INT_PRINTF(("#### HDMI/DVI Mode : DVI ####\n"));
	}
}


static unsigned char CLKCheck(unsigned char ucPortSel)
{
	unsigned char sys_state;
	if(ucPortSel == 1)
	{
		Read_Ite6801_i2c(ITE_HDMI_I2C_ADDR, REG_RX_0B, &sys_state);
		sys_state |= 0x08;
		//sys_state = hdmirxrd(REG_RX_P1_SYS_STATUS) & (B_P1_RXCK_VALID);
	}
	else
	{
		Read_Ite6801_i2c(ITE_HDMI_I2C_ADDR, REG_RX_0A, &sys_state);
		sys_state |= B_P0_SCDT;
		//sys_state = hdmirxrd(REG_RX_P0_SYS_STATUS) & (B_P0_RXCK_VALID);
	}
	if(sys_state == 0x08)
		return TRUE;
	else
		return FALSE;
}


static void hdmirx_INT_P0_ECC(void)
{
	//struct it6802_dev_data *it6802data = get_it6802_dev_data();
	uint08 rddata;

	if((m_ucEccCount_P0++) > 21)
	{

		//if(it6802->EQPort[F_PORT_SEL_0].f_manualEQadjust==TRUE)	// ignore ECC interrupt when manual EQ adjust !!!
		//return;

		m_ucEccCount_P0=0;

		//IT6802_DEBUG_INT_PRINTF(("CDR reset for Port0 ECC_TIMEOUT !!!\n"));
		Read_Ite6801_i2c(ITE_HDMI_I2C_ADDR, REG_RX_0A, &rddata);

		if (rddata |= B_P0_MHL_MODE)//if((hdmirxrd(0x0A)&0x40))
		{
			Ite6801RegSet(MHL_ADDR, REG_RX_28, 0x40,0x40 );//mhlrxset(MHL_RX_28,0x40,0x40);
			//it6802HPDCtrl(0,1);
			__delay_cycles(100000);//delay1ms(100);
			//it6802HPDCtrl(0,0);
			Ite6801RegSet(MHL_ADDR, REG_RX_28, 0x40,0x00 );//mhlrxset(MHL_RX_28,0x40,0x00);

		}
		else
		{
			it6801HPDCtrl(0,0);	// MHL port , set HPD = 0
		}
		Ite6801RegSet(ITE_HDMI_I2C_ADDR, REG_RX_11, 0x0d,0x0d );
		//hdmirxset(REG_RX_011,(B_P0_DCLKRST|B_P0_CDRRST|B_P0_SWRST),(B_P0_DCLKRST|B_P0_CDRRST|B_P0_SWRST));
		__delay_cycles(300000);//delay1ms(300);
		Ite6801RegSet(ITE_HDMI_I2C_ADDR, REG_RX_11, 0x0d,0x00 );
		//hdmirxset(REG_RX_011,(B_P0_DCLKRST|B_P0_CDRRST|B_P0_SWRST),0x00);

		IT6801SwitchVideoState(VSTATE_SyncWait);

		//set port 0 HPD=1
		it6801HPDCtrl(0,1);	// MHL port , set HPD = 1
	}
}

static void TMDSCheck(unsigned char ucPortSel)//Auto EQ undo
{

}
static unsigned char CheckPlg5VPwr(unsigned char ucPortSel)
{
	unsigned char sys_state_P0;
	unsigned char sys_state_P1;

	if(ucPortSel==0)
	{
		Read_Ite6801_i2c(ITE_HDMI_I2C_ADDR, REG_RX_0A, &sys_state_P0);//sys_state_P0 = hdmirxrd(REG_RX_P0_SYS_STATUS) & B_P0_PWR5V_DET;

		if((sys_state_P0 & B_P0_PWR5V_DET))
		{

				//chgbank(0);
				//reg0Ah = hdmirxrd(0x0A);
				//if( (reg0Ah&0x40) == 0)
				//BUSMODE = MHL;
				//else
				//BUSMODE = HDMI;
				 //if( BUSMODE==HDMI ) {


//2013-08-01 disable -->if((hdmirxrd(0x0A)&0x40)==0){
//2013-08-01 disable -->chgbank(1);
//2013-08-01 disable -->hdmirxset(REG_RX_1B0, 0x03, 0x03);
//2013-08-01 disable -->chgbank(0);
//2013-08-01 disable -->}

//xxxxx 2013-0801
			it6801HPDCtrl(0,1);	// MHL port , set HPD = 1
//xxxxx


			return TRUE;
		}
		else
		{
//xxxxx 2013-0801
			it6801HPDCtrl(0,0);	// MHL port , set HPD = 0
//xxxxx


			return FALSE;
		}
	}
	else
	{
		Read_Ite6801_i2c(ITE_HDMI_I2C_ADDR, REG_RX_0B, &sys_state_P1);//sys_state_P1 = hdmirxrd(REG_RX_P1_SYS_STATUS) & B_P1_PWR5V_DET;
		if((sys_state_P1 & 0x01))
		{

			//xxxxx 2013-0801
			//			HotPlug(1);	//set port 1 HPD=1
			//xxxxx
			//xxxxx 2013-0801
				it6801HPDCtrl(1,1);	// HDMI port , set HPD = 1
			//xxxxx

			return TRUE;
		}
		else
		{
			//xxxxx 2013-0801
			//			HotPlug(0);	//set port 1 HPD=0
			//xxxxx
			//xxxxx 2013-0801
				it6801HPDCtrl(1,0);	// HDMI port , set HPD = 0
			//xxxxx

			return FALSE;
		}

	}
}

static void hdmirx_INT_5V_Pwr_Chg(unsigned char ucport)
{

	unsigned char ucPortSel;
	//ucPortSel = hdmirxrd(REG_RX_051)&B_PORT_SEL;
	Read_Ite6801_i2c(ITE_HDMI_I2C_ADDR, REG_RX_51, &ucPortSel);
	ucPortSel &=B_PORT_SEL ;

	if(ucPortSel == ucport)
	{
		if(CheckPlg5VPwr(ucport)){
			//IT6802_DEBUG_INT_PRINTF(("#### Power 5V ON ####\n"));
			IT6801SwitchVideoState(VSTATE_SyncWait);
		}
		else
		{
			//IT6802_DEBUG_INT_PRINTF(("#### Power 5V OFF ####\n"));
			IT6801SwitchVideoState(VSTATE_SWReset);
		}
	}

}
static unsigned char CheckSCDT(void)
{
	unsigned char ucPortSel;
	unsigned char sys_state_P0;

	Read_Ite6801_i2c(ITE_HDMI_I2C_ADDR, REG_RX_51, &ucPortSel);//ucPortSel = hdmirxrd(REG_RX_051) & B_PORT_SEL;
	ucPortSel &= 0x01;

	Read_Ite6801_i2c(ITE_HDMI_I2C_ADDR, REG_RX_0A, &sys_state_P0);//sys_state_P0=hdmirxrd(REG_RX_P0_SYS_STATUS);

	if(ucPortSel == m_ucCurrentHDMIPort)
	{

		if(sys_state_P0 & B_DisPixRpt )
		{
			//SCDT on
			//it6802->m_ucSCDTOffCount=0;
			return TRUE;
		}
		else
		{
			//SCDT off
			return FALSE;
		}

	}
	return FALSE;
}

static void hdmirx_INT_SCDT_Chg(void)
{
	if(CheckSCDT() == TRUE){
		//IT6802_DEBUG_INT_PRINTF(("#### SCDT ON ####\n"));
		IT6801SwitchVideoState(VSTATE_SyncChecking);
	}
	else{
		//IT6802_DEBUG_INT_PRINTF(("#### SCDT OFF ####\n"));
		IT6801SwitchVideoState(VSTATE_SyncWait);
		//IT6802SwitchAudioState(it6802,ASTATE_AudioOff);//remark June

//		TMDSCheck(it6802->m_ucCurrentHDMIPort);
//		TogglePolarity (it6802->m_ucCurrentHDMIPort);


	}
}

static void IT6801HDMIInterruptHandler(void)
{
	unsigned char Reg05h;
	unsigned char Reg06h;
	unsigned char Reg07h;
	unsigned char Reg08h;
	unsigned char Reg09h;
	unsigned char Reg0Ah;
//	unsigned char Reg0Bh;
	unsigned char RegD0h;

	Ite6801RegSet(ITE_HDMI_I2C_ADDR,REG_RX_0F,0x03,BANK0);//chgbank(0);

	Read_Ite6801_i2c(ITE_HDMI_I2C_ADDR, REG_RX_05, &Reg05h);//Reg05h = hdmirxrd(REG_RX_005);
	Read_Ite6801_i2c(ITE_HDMI_I2C_ADDR, REG_RX_06, &Reg06h);//Reg06h = hdmirxrd(REG_RX_006);
	Read_Ite6801_i2c(ITE_HDMI_I2C_ADDR, REG_RX_07, &Reg07h);//Reg07h = hdmirxrd(REG_RX_007);
	Read_Ite6801_i2c(ITE_HDMI_I2C_ADDR, REG_RX_08, &Reg08h);//Reg08h = hdmirxrd(REG_RX_008);
	Read_Ite6801_i2c(ITE_HDMI_I2C_ADDR, REG_RX_09, &Reg09h);//Reg09h = hdmirxrd(REG_RX_009);

	Read_Ite6801_i2c(ITE_HDMI_I2C_ADDR, REG_RX_0A, &Reg0Ah);//Reg0Ah = hdmirxrd(REG_RX_P0_SYS_STATUS);
//	Reg0Bh = hdmirxrd(REG_RX_P1_SYS_STATUS);
	Read_Ite6801_i2c(ITE_HDMI_I2C_ADDR, REG_RX_0D, &RegD0h);//RegD0h = hdmirxrd(REG_RX_0D0);

	Ite6801RegSet(ITE_HDMI_I2C_ADDR, REG_RX_05, 0xFF,Reg05h );//hdmirxwr(REG_RX_005, Reg05h);
	Ite6801RegSet(ITE_HDMI_I2C_ADDR, REG_RX_06, 0xFF,Reg06h );//hdmirxwr(REG_RX_006, Reg06h);
	Ite6801RegSet(ITE_HDMI_I2C_ADDR, REG_RX_07, 0xFF,Reg07h );//hdmirxwr(REG_RX_007, Reg07h);
	Ite6801RegSet(ITE_HDMI_I2C_ADDR, REG_RX_08, 0xFF,Reg08h );//hdmirxwr(REG_RX_008, Reg08h);
	Ite6801RegSet(ITE_HDMI_I2C_ADDR, REG_RX_09, 0xFF,Reg09h );//hdmirxwr(REG_RX_009, Reg09h);
//2013-0606 disable ==>
	Ite6801RegSet(ITE_HDMI_I2C_ADDR, REG_RX_0D, 0xFF,(RegD0h&0x0F) );//hdmirxwr(REG_RX_0D0, RegD0h&0x0F);


//	IT6802_DEBUG_INT_PRINTF(("111111111111111111111111 STATUS 111111111111111111111= %X \r\n",hdmirxrd(REG_RX_P0_SYS_STATUS)));
     if( Reg05h!=0x00 )
	{

		//IT6802_DEBUG_INT_PRINTF(("Reg05 = %X \r\n",(int) Reg05h));

		 if( Reg05h&0x80 )
		 {
			 //IT6802_DEBUG_INT_PRINTF(("#### Port 0 HDCP Off Detected ###\n"));
			m_ucEccCount_P0=0;
		 }

		 if( Reg05h&0x40 )
		 {
			// IT6802_DEBUG_INT_PRINTF(("#### Port 0 ECC Error %X ####\n",(int) (it6802->m_ucEccCount_P0)));
//			HDMICheckErrorCount(&(it6802->EQPort[F_PORT_SEL_0]));	//07-04 for port 0
			hdmirx_INT_P0_ECC();
		 }

		 if( Reg05h&0x20 )
		 {

		     //IT6802_DEBUG_INT_PRINTF(("#### Port 0 HDMI/DVI Mode change ####\n"));
//FIX_ID_009 xxxxx	//verify interrupt event with reg51[0] select port
			if(CLKCheck(0))
			hdmirx_INT_HDMIMode_Chg(0);
//FIX_ID_009 xxxxx

		 }

		 if( Reg05h&0x08 )
		 {
			 //IT6802_DEBUG_INT_PRINTF(("#### Port 0 HDCP Authentication Start ####\n"));
			m_ucEccCount_P0=0;
//			get_vid_info();
//			show_vid_info();

//FIX_ID_014 xxxxx
			if( ucPortAMPOverWrite[0] == 0)
			{
				if(( HDMIIntEvent & 0x01 )==0)
				{
					//hdmirxwr(REG_RX_022, 0x00);	// power down auto EQ

					HDMIIntEvent |= 0x01;
					HDMIIntEvent |= 0x02;
					HDMIWaitNo[0]=3;
				}
				else if((HDMIIntEvent & (0x02)))
				{
					HDMIIntEvent |= 0x01;
					HDMIWaitNo[0] += 1;
				}
			}
			else
			{
				if(HDMIIntEvent & 0x02)
				{
					HDMIIntEvent |= 0x01;
					HDMIWaitNo[0] += 1;
				}
			}
//FIX_ID_014 xxxxx

//FIX_ID_005 xxxxx	//for waiting RAP content on
			if( (Reg0Ah&0x40))
			{
				CBusIntEvent |= 0x10;
				CBusWaitNo=2;
			}
//FIX_ID_005 xxxxx


		 }

		 if( Reg05h&0x10 )
		 {
			 //IT6802_DEBUG_INT_PRINTF(("#### Port 0 HDCP Authentication Done ####\n"));
//FIX_ID_005 xxxxx	//for waiting RAP content on
			if( (Reg0Ah&0x40))
			{
				CBusIntEvent |= 0x10;
				CBusWaitNo=2;
			}
//FIX_ID_005 xxxxx

//FIX_ID_014 xxxxx
			/*if((it6802->HDMIIntEvent & (B_PORT0_Waiting)))
			{
				it6802->HDMIWaitNo[0] = 0;
			}*///remark June
//FIX_ID_014 xxxxx

		 }

		 if( Reg05h&0x04 )
		 {
			 //IT6802_DEBUG_INT_PRINTF(("#### Port 0 Input Clock Change Detect ####\n"));
		 }

		 if( Reg05h&0x02 )
		 {

			m_ucEccCount_P0=0;
			//m_ucDeskew_P0=0;
			//it6802->m_ucDeskew_P1=0;
			//it6802->m_ucEccCount_P1=0;

			//IT6802_DEBUG_INT_PRINTF(("#### Port 0 Rx CKOn Detect ####\n"));

			// NO --> Authentication Start 	&& 	Input Clock Change Detect 	&&	 B_PORT1_TMDSEvent
			if(( Reg05h&0x08 )==0 && ( Reg05h&0x04 )==0  &&  (HDMIIntEvent & 0x02)==0)
			{
					if(CLKCheck(0))
					{
						TMDSCheck(0);
					}
			}
			else
			{
				if(( Reg05h&0x10 ) == 0)
				{
					if((HDMIIntEvent & 0x01)==0)
					{
						//hdmirxwr(REG_RX_022, 0x00);	// power down auto EQ
						HDMIIntEvent |= 0x01;
						HDMIIntEvent |= 0x02;
						HDMIWaitNo[0]=3;
					}
				}
				else
				{
					if(CLKCheck(0))
					{
						TMDSCheck(0);
					}
				}
			}
		 }

		 if( Reg05h&0x01 )
		 {
		//	IT6802_DEBUG_INT_PRINTF(("#### Port 0 Power 5V change ####\n"));
			hdmirx_INT_5V_Pwr_Chg(0);


			//FIX_ID_001 xxxxx Add Auto EQ with Manual EQ
			if(CheckPlg5VPwr(0)==FALSE)
			{

				#ifdef _SUPPORT_EQ_ADJUST_
				DisableOverWriteRS(F_PORT_SEL_0);
				#endif
			}
			//FIX_ID_001 xxxxx

		 }
	 }
#if 0
     if( Reg06h!=0x00 )
	 {
		 if( Reg06h&0x80 )
		 {
			//IT6802_DEBUG_INT_PRINTF(("#### Port 1 HDCP Off Detected ###\n"));
			m_ucEccCount_P1=0;

		 }

		 if( Reg06h&0x40 )
		 {
			 //IT6802_DEBUG_INT_PRINTF(("#### Port 1 ECC Error ####\n"));
			hdmirx_INT_P1_ECC(it6802);
		 }

		 if( Reg06h&0x20 )
		 {
		     IT6802_DEBUG_INT_PRINTF(("#### Port 1 HDMI/DVI Mode change ####\n"));
//FIX_ID_009 xxxxx	//verify interrupt event with reg51[0] select port
			if(CLKCheck(1))
			hdmirx_INT_HDMIMode_Chg(it6802,1);
//FIX_ID_009 xxxxx
		 }

		 if( Reg06h&0x10 )
		 {
			 IT6802_DEBUG_INT_PRINTF(("#### Port 1 HDCP Authentication Done ####\n"));
//FIX_ID_014 xxxxx
			if((it6802->HDMIIntEvent & (B_PORT1_Waiting)))
			{
				it6802->HDMIWaitNo[1] = 0;
			}
//FIX_ID_014 xxxxx

		 }

		 if( Reg06h&0x08 )
		 {
			 IT6802_DEBUG_INT_PRINTF(("#### Port 1 HDCP Authentication Start ####\n"));
			it6802->m_ucEccCount_P1=0;

//FIX_ID_014 xxxxx
			if( ucPortAMPOverWrite[F_PORT_SEL_1] == 0)
			{
				if((it6802->HDMIIntEvent & (B_PORT1_Waiting))==0)
				{
					hdmirxwr(REG_RX_03A, 0x00);	// power down auto EQ

					it6802->HDMIIntEvent |= (B_PORT1_Waiting);
					it6802->HDMIIntEvent |= (B_PORT1_TMDSEvent);
					it6802->HDMIWaitNo[1]=MAX_TMDS_WAITNO;
				}
				else if((it6802->HDMIIntEvent & (B_PORT1_TMDSEvent)))
				{
					it6802->HDMIIntEvent |= (B_PORT1_Waiting);
					it6802->HDMIWaitNo[1] += MAX_HDCP_WAITNO;
				}
			}
			else
			{
				if((it6802->HDMIIntEvent & (B_PORT1_TMDSEvent)))
				{
					it6802->HDMIIntEvent |= (B_PORT1_Waiting);
					it6802->HDMIWaitNo[1] += MAX_HDCP_WAITNO;
				}
			}
//FIX_ID_014 xxxxx

		 }

		 if( Reg06h&0x04 )
		 {
			 IT6802_DEBUG_INT_PRINTF(("#### Port 1 Input Clock Change Detect ####\n"));
		 }

		 if( Reg06h&0x02 )
		 {
			IT6802_DEBUG_INT_PRINTF(("#### Port 1 Rx CKOn Detect ####\n"));
			//it6802->m_ucEccCount_P0=0;
			//it6802->m_ucDeskew_P0=0;
			it6802->m_ucDeskew_P1=0;
			it6802->m_ucEccCount_P1=0;

			// NO --> Authentication Start 	&& 	Input Clock Change Detect 	&&	 B_PORT1_TMDSEvent
			if(( Reg06h&0x08 )==0 && ( Reg06h&0x04 )==0  &&  (it6802->HDMIIntEvent & (B_PORT1_TMDSEvent))==0)
			{
					if(CLKCheck(F_PORT_SEL_1))
					{
						TMDSCheck(F_PORT_SEL_1);
					}
			}
			else
			{
				if(( Reg06h&0x10 ) == 0)
				{
					if((it6802->HDMIIntEvent & (B_PORT1_Waiting))==0)
					{
						hdmirxwr(REG_RX_03A, 0x00);	// power down auto EQ
						it6802->HDMIIntEvent |= (B_PORT1_Waiting);
						it6802->HDMIIntEvent |= (B_PORT1_TMDSEvent);
						it6802->HDMIWaitNo[1]=MAX_TMDS_WAITNO;
					}
				}
				else
				{
					if(CLKCheck(F_PORT_SEL_1))
					{
						TMDSCheck(F_PORT_SEL_1);
					}
				}
			}



		}



		 if( Reg06h&0x01 )
		 {
			IT6802_DEBUG_INT_PRINTF(("#### Port 1 Power 5V change ####\n"));
			hdmirx_INT_5V_Pwr_Chg(it6802,1);
			//FIX_ID_001 xxxxx Add Auto EQ with Manual EQ
			if(CheckPlg5VPwr(F_PORT_SEL_1)==FALSE)
			{
				#ifdef _SUPPORT_EQ_ADJUST_
				DisableOverWriteRS(F_PORT_SEL_1);
				#endif
			}
			//FIX_ID_001 xxxxx
	 	}

     	}
#endif
	 if( Reg07h!=0x00)
	 {
#if 0
		 if( Reg07h&0x80 ) {
			 IT6802_DEBUG_INT_PRINTF(("#### Audio FIFO Error ####\n"));
			 aud_fiforst();
		 }

		 if( Reg07h&0x40 ) {
			 IT6802_DEBUG_INT_PRINTF(("#### Audio Auto Mute ####\n"));
		 }

		 if( Reg07h&0x20 ) {
			 IT6802_DEBUG_INT_PRINTF(("#### Packet Left Mute ####\n"));
			 IT6802_SetVideoMute(it6802,OFF);
		 }

		 if( Reg07h&0x10 ) {
			 IT6802_DEBUG_INT_PRINTF(("#### Set Mute Packet Received ####\n"));

			 IT6802_SetVideoMute(it6802,ON);
		 }

		 if( Reg07h&0x08 ) {
			 IT6802_DEBUG_INT_PRINTF(("#### Timer Counter Tntterrupt ####\n"));
			//if(it6802->m_VState == VSTATE_VideoOn)
			//	hdmirxset(0x84,0x80,0x80);	//2011/06/17 xxxxx, for enable Rx Chip count

		 }

		 if( Reg07h&0x04 ) {
			 IT6802_DEBUG_INT_PRINTF(("#### Video Mode Changed ####\n"));
		 }
#endif

		 if( Reg07h&0x02 )
		 {
			hdmirx_INT_SCDT_Chg();
		 }

		 if( Reg07h&0x01 )
		 {
			 if( (Reg0Ah&0x40)>>6 )
			 {
				 //IT6802_DEBUG_INT_PRINTF(("#### Port 0 Bus Mode : MHL ####\n"));

				//FIX_ID_002 xxxxx 	Check IT6802 chip version Identify for TogglePolarity and Port 1 Deskew
				/*if(HdmiI2cAddr==IT6802A0_HDMI_ADDR)
				{
					chgbank(1);
					hdmirxset(REG_RX_1B6,0x07,0x00);
					//FIX_ID_007 xxxxx 	//for debug IT6681  HDCP issue
					hdmirxset(REG_RX_1B1,0x20,0x20);//Reg1b1[5] = 1 for enable over-write
					hdmirxset(REG_RX_1B2,0x07,0x01);	// default 0x04 , change to 0x01
					IT6802_DEBUG_INT_PRINTF((" Port 0 Bus Mode Reg1B1  = %X ,Reg1B2  = %X\r\n",(int) hdmirxrd(REG_RX_1B1),(int) hdmirxrd(REG_RX_1B2)));
					//FIX_ID_007 xxxxx
					chgbank(0);
				}*/ //remark June slave address = 0x90 != IT6802A0_HDMI_ADDR
				//FIX_ID_002 xxxxx

				it6801HPDCtrl(0,1);	// MHL port , set HPD = 1

			 }
			 /*else
			 {
				//IT6802_DEBUG_INT_PRINTF(("#### Port 0 Bus Mode : HDMI ####\n"));
				//FIX_ID_002 xxxxx 	Check IT6802 chip version Identify for TogglePolarity and Port 1 Deskew
					if(HdmiI2cAddr==IT6802A0_HDMI_ADDR)
					{
						chgbank(1);
						hdmirxset(REG_RX_1B6,0x07,0x03);
						////FIX_ID_007 xxxxx 	//for debug IT6681  HDCP issue
						hdmirxset(REG_RX_1B1,0x20,0x00);//Reg1b1[5] = 0 for disable over-write
						hdmirxset(REG_RX_1B2,0x07,0x04);	// default 0x04 , change to 0x01
						EQ_PORT0_PRINTF((" Port 0 Bus Mode Reg1B1  = %X ,Reg1B2  = %X\r\n",(int) hdmirxrd(REG_RX_1B1),(int) hdmirxrd(REG_RX_1B2)));
						////FIX_ID_007 xxxxx
						chgbank(0);
					}
				//FIX_ID_002 xxxxx

			 }*/ //remark June slave address = 0x90 != IT6802A0_HDMI_ADDR

		 }
	 }

	 /*if( Reg08h != 0x00)
	 {
		 if( Reg08h&0x80 ) {
			//			 MHLRX_DEBUG_PRINTF(("#### No General Packet 2 Received ####\n"));
		 }

		 if( Reg08h&0x40 ) {
			//			 MHLRX_DEBUG_PRINTF(("#### No General Packet Received ####\n"));
		 }

		 if( Reg08h&0x20 ) {
			// IT6802_DEBUG_INT_PRINTF(("#### No Audio InfoFrame Received ####\n"));
		 }

		 if( Reg08h&0x10) {
			// IT6802_DEBUG_INT_PRINTF(("#### No AVI InfoFrame Received ####\n"));
		 }

		 if( Reg08h&0x08 ) {
			// IT6802_DEBUG_INT_PRINTF(("#### CD Detect ####\n"));

		 }

		 if( Reg08h&0x04 ) {
			//			 MHLRX_DEBUG_PRINTF(("#### Gen Pkt Detect ####\n"));
			 //IT6802_DEBUG_INT_PRINTF(("#### 3D InfoFrame Detect ####\n"));

				#ifdef Enable_Vendor_Specific_packet
					if(it6802->f_de3dframe_hdmi == FALSE)
					{
					it6802->f_de3dframe_hdmi = IT6802_DE3DFrame(TRUE);
					}
				#endif

		 }

		 if( Reg08h&0x02 ) {
			 //IT6802_DEBUG_INT_PRINTF(("#### ISRC2 Detect ####\n"));
		 }

		 if( Reg08h&0x01 ) {
			// IT6802_DEBUG_INT_PRINTF(("#### ISRC1 Detect ####\n"));
		 }
	 }*/ //remark June

	 if( Reg09h!=0x00 )
	 {
        	 if( Reg09h&0x80 )
		{
			 //IT6802_DEBUG_INT_PRINTF(("#### H2V Buffer Skew Fail ####\n"));
		 }

		 if( Reg09h&0x40 )
		 {

			//FIX_ID_002 xxxxx 	Check IT6802 chip version Identify for TogglePolarity and Port 1 Deskew
			/*if(HdmiI2cAddr==IT6802A0_HDMI_ADDR)
			{
				hdmirxwr(0x09, 0x20); //bug ~ need to update by Andrew
			}
			else
			{
				hdmirxwr(0x09, 0x40);
			}*/
			 Ite6801RegSet(ITE_HDMI_I2C_ADDR, REG_RX_09, 0xFF,0x40 );
			//FIX_ID_002 xxxxx
			//IT6802_DEBUG_INT_PRINTF(("#### Port 1 Deskew Error ####\n"));
			//hdmirx_INT_P1_Deskew(it6802);//remark June EQ???
		 }

		 if( Reg09h&0x20 ) {
			 Ite6801RegSet(ITE_HDMI_I2C_ADDR, REG_RX_09, 0xFF,0x20 );//hdmirxwr(0x09, 0x20);
			 //IT6802_DEBUG_INT_PRINTF(("#### Port 0 Deskew Error ####\n"));
			//hdmirx_INT_P0_Deskew(it6802);//remark June EQ ???
		 }

		 if( Reg09h&0x10 ) {
			// IT6802_DEBUG_INT_PRINTF(("#### New Audio Packet Received ####\n"));
		 }

		 if( Reg09h&0x08 ) {
			 //IT6802_DEBUG_INT_PRINTF(("#### New ACP Packet Received ####\n"));
		 }

		 if( Reg09h&0x04 ) {
			 //IT6802_DEBUG_INT_PRINTF(("#### New SPD Packet Received ####\n"));
		 }

		 if( Reg09h&0x02) {
			// IT6802_DEBUG_INT_PRINTF(("#### New MPEG InfoFrame Received ####\n"));
		 }

		 if( Reg09h&0x01) {
			 //IT6802_DEBUG_INT_PRINTF(("#### New AVI InfoFrame Received ####\n"));
			//IT6802VideoOutputConfigure();
			m_NewAVIInfoFrameF=TRUE;
		 }

	 }


	if( RegD0h!=0x00 )
	{
// disable		if( RegD0h&0x08)
// disable		{
// disable			EQ_DEBUG_PRINTF(("#### Port 1 Rx Clock change detect Interrupt ####\n"));
// disable		}
// disable
// disable		if( RegD0h&0x04)
// disable		{
// disable			EQ_DEBUG_PRINTF(("#### Port 0 Rx Clock change detect Interrupt ####\n"));
// disable		}
	/* if( RegD0h&0x10 )
	 {

		hdmirxwr(0xD0, 0x30);
		RegD0h&=0x30;
		ucEqRetryCnt[0]=0;
		 EQ_PORT0_PRINTF(("#### Port 0 EQ done interrupt ####\n"));

	//2013-0923 disable ->	ucPortAMPOverWrite[0]=1;	//2013-0801
		AmpValidCheck(0);	//2013-0801


//FIX_ID_001 xxxxx Add Auto EQ with Manual EQ
	#ifdef _SUPPORT_EQ_ADJUST_
	HDMIStartEQDetect(&(it6802->EQPort[F_PORT_SEL_0]));
	#endif
//FIX_ID_001 xxxxx

 }*/ //remark June

	/* if( RegD0h&0x40 )
	 {

	hdmirxwr(0xD0, 0xC0);
	RegD0h&=0xC0;
	ucEqRetryCnt[1]=0;
	// EQ_PORT1_PRINTF(("#### Port 1 EQ done interrupt ####\n"));


//2013-0923 disable ->	ucPortAMPOverWrite[1]=1;	//2013-0801
	AmpValidCheck(1);	//2013-0801


//FIX_ID_001 xxxxx Add Auto EQ with Manual EQ
	#ifdef _SUPPORT_EQ_ADJUST_
	HDMIStartEQDetect(&(it6802->EQPort[F_PORT_SEL_1]));
	#endif
//FIX_ID_001 xxxxx
 }*///remark June

	/*if( RegD0h&0x20)
	{

	hdmirxwr(0xD0, 0x20);
	//EQ_PORT0_PRINTF(("#### Port 0 EQ Fail Interrupt ####\n"));
//	HDMICheckErrorCount(&(it6802->EQPort[F_PORT_SEL_0]));	//07-04 for port 0
//FIX_ID_001 xxxxx Add Auto EQ with Manual EQ
	#ifdef _SUPPORT_AUTO_EQ_
	hdmirx_INT_EQ_FAIL(it6802,F_PORT_SEL_0);
	#endif
//FIX_ID_001 xxxxx
}*///remark June

	/*if( RegD0h&0x80)
		{

	hdmirxwr(0xD0, 0x80);
	EQ_PORT1_PRINTF(("#### Port 1 EQ Fail Interrupt ####\n"));
//	HDMICheckErrorCount(&(it6802->EQPort[F_PORT_SEL_1]));	//07-04 for port 0
//FIX_ID_001 xxxxx Add Auto EQ with Manual EQ
	#ifdef _SUPPORT_AUTO_EQ_
	hdmirx_INT_EQ_FAIL(it6802,F_PORT_SEL_1);
	#endif
//FIX_ID_001 xxxxx
}*/ //remark June



	}

}

static void WaitingForSCDT(void)
{
	unsigned char sys_state_P0;
	unsigned char sys_state_P1;
	unsigned char ucPortSel;
//	unsigned char ucTMDSClk ;

	Read_Ite6801_i2c(ITE_HDMI_I2C_ADDR, REG_RX_0A, &sys_state_P0);//sys_state_P0=hdmirxrd(REG_RX_P0_SYS_STATUS) & (B_P0_SCDT|B_P0_PWR5V_DET|B_P0_RXCK_VALID);
	sys_state_P0 &= 0x89;
	Read_Ite6801_i2c(ITE_HDMI_I2C_ADDR, REG_RX_0B, &sys_state_P1);//sys_state_P1=hdmirxrd(REG_RX_P1_SYS_STATUS) & (B_P1_SCDT|B_P1_PWR5V_DET|B_P1_RXCK_VALID);
	sys_state_P1 &=0x89;
	Read_Ite6801_i2c(ITE_HDMI_I2C_ADDR, REG_RX_51, &ucPortSel);//ucPortSel = hdmirxrd(REG_RX_051) & B_PORT_SEL;
	ucPortSel &=0x01;

	if(sys_state_P0 & 0x80)
	{
		IT6801SwitchVideoState(VSTATE_SyncChecking);	//2013-0520
		return;
	}
	/*else
	{
		if(it6802->EQPort[ucPortSel].f_manualEQadjust==TRUE)		// ignore SCDT off when manual EQ adjust !!!
		{
			return;
		}


		if(ucPortSel == F_PORT_SEL_0)
		{

			if((sys_state_P0 & (B_P0_PWR5V_DET|B_P0_RXCK_VALID)) == (B_P0_PWR5V_DET|B_P0_RXCK_VALID))
			{
				it6802->m_ucSCDTOffCount++;
					EQ_PORT0_PRINTF((" SCDT off count = %X \r\n",(int)it6802->m_ucSCDTOffCount));
					EQ_PORT0_PRINTF((" sys_state_P0 = %X \r\n",(int)hdmirxrd(REG_RX_P0_SYS_STATUS)));

			}
		}
		else
		{
			if((sys_state_P1 & (B_P1_PWR5V_DET|B_P1_RXCK_VALID)) == (B_P1_PWR5V_DET|B_P1_RXCK_VALID))
			{
				it6802->m_ucSCDTOffCount++;
					EQ_PORT1_PRINTF((" SCDT off count = %X \r\n",(int)it6802->m_ucSCDTOffCount));
					EQ_PORT1_PRINTF((" sys_state_P1 = %X \r\n",(int)hdmirxrd(REG_RX_P1_SYS_STATUS)));

			}
		}

		if((it6802->m_ucSCDTOffCount)>SCDT_OFF_TIMEOUT)
		{
			it6802->m_ucSCDTOffCount=0;


			if(ucPortSel == F_PORT_SEL_0)
				{

					hdmirxset(REG_RX_011,(B_P0_DCLKRST|B_P0_CDRRST),(B_P0_DCLKRST|B_P0_CDRRST|B_P0_SWRST));
					hdmirxset(REG_RX_011,(B_P0_DCLKRST|B_P0_CDRRST),0x00);
					EQ_PORT0_PRINTF((" WaitingForSCDT( ) Port 0 CDR reset !!! \r\n"));

//xxxxx
					DisableOverWriteRS(0);
					TMDSCheck(0);
//xxxxx



				}
			else
				{
					hdmirxset(REG_RX_018,(B_P1_DCLKRST|B_P1_CDRRST),(B_P1_DCLKRST|B_P1_CDRRST|B_P1_SWRST));
					hdmirxset(REG_RX_018,(B_P1_DCLKRST|B_P1_CDRRST),0x00);
					EQ_PORT1_PRINTF((" WaitingForSCDT( ) Port 1 CDR reset !!! \r\n"));

//xxxxx
					DisableOverWriteRS(1);
					TMDSCheck(1);
//xxxxx

				}

		}

	}*/
}

static unsigned char CheckAVMute(void)
{

	unsigned char ucAVMute;
	unsigned char ucPortSel;

	Read_Ite6801_i2c(ITE_HDMI_I2C_ADDR, REG_RX_A8, &ucAVMute);//ucAVMute=hdmirxrd(REG_RX_0A8) & (B_P0_AVMUTE|B_P1_AVMUTE);
	ucAVMute &=0x11;
	Read_Ite6801_i2c(ITE_HDMI_I2C_ADDR, REG_RX_51, &ucPortSel);//ucPortSel = hdmirxrd(REG_RX_051)&B_PORT_SEL;
	ucPortSel &=0x01;

	if(((ucAVMute & 0x01)&& (ucPortSel == 0 ))||
	((ucAVMute & 0x10)&& (ucPortSel == 1 )))
	{
		return TRUE;
	}
	else
	{
		return FALSE;
	}

}

static void IT6801_SetVideoMute(unsigned char bMute)
{

    if(bMute)
    {
    	//******** AV Mute -> ON ********//
    	Ite6801RegSet(ITE_HDMI_I2C_ADDR, REG_RX_53, 0xC0,0xC0 );//hdmirxset(REG_RX_053,(B_VDGatting|B_VIOSel),(B_VDGatting|B_VIOSel));	//Reg53[7][5] = 11    // for enable B_VDIO_GATTING and VIO_SEL
    	Ite6801RegSet(ITE_HDMI_I2C_ADDR, REG_RX_52, 0x20,0x20 );//hdmirxset(REG_RX_052,(B_DisVAutoMute),(B_DisVAutoMute));				//Reg52[5] = 1 for disable Auto video MUTE
    	Ite6801RegSet(ITE_HDMI_I2C_ADDR, REG_RX_53, 0x0E,0x00 );//hdmirxset(REG_RX_053,(B_TriVDIO),(0x00));								//Reg53[2:0] = 000;         // 0 for enable video io data output

    	//HDMIRX_AUDIO_PRINTF(("+++++++++++ IT6802_SetVideoMute -> On +++++++++++++++++\n"));
    }
    else
    {
        if(Cur_VSTATE == VSTATE_VideoOn)
        {
        	//******** AV Mute -> OFF ********//
            if(CheckAVMute()==TRUE)
            {
            	Ite6801RegSet(ITE_HDMI_I2C_ADDR, REG_RX_52, 0x20,0x20 );//hdmirxset(REG_RX_052,(B_DisVAutoMute),(B_DisVAutoMute));				//Reg52[5] = 1 for disable Auto video MUTE
            }
            else
            {
            	Ite6801RegSet(ITE_HDMI_I2C_ADDR, REG_RX_53, 0x0E,0x0E );//hdmirxset(REG_RX_053,(B_TriVDIO),(B_TriVDIO));							//Reg53[2:0] = 111;         // 1 for enable tri-state of video io data
            	Ite6801RegSet(ITE_HDMI_I2C_ADDR, REG_RX_53, 0x0E,0x00 );//hdmirxset(REG_RX_053,(B_TriVDIO),(0x00));								//Reg53[2:0] = 000;         // 0 for enable video io data output

            	Ite6801RegSet(ITE_HDMI_I2C_ADDR, REG_RX_53, 0xC0,0xC0 );//hdmirxset(REG_RX_053,(B_VDGatting|B_VIOSel),(B_VDGatting|B_VIOSel));	//Reg53[7][5] = 11    // for enable B_VDIO_GATTING and VIO_SEL
            	Ite6801RegSet(ITE_HDMI_I2C_ADDR, REG_RX_53, 0xC0,0x40 );//hdmirxset(REG_RX_053,(B_VDGatting|B_VIOSel),(B_VIOSel));				//Reg53[7][5] = 01    // for disable B_VDIO_GATTING
            	//HDMIRX_AUDIO_PRINTF(("+++++++++++  IT6802_SetVideoMute -> Off +++++++++++++++++\n"));
            }

        }

    }

}



static void IT6801VideoHandler(void)
{
//	unsigned char uc;
	uint08 rxdata;

	/*if(it6802->m_VideoCountingTimer > MS_LOOP)
	{
	it6802->m_VideoCountingTimer -= MS_LOOP;
	}
	else
	{
	it6802->m_VideoCountingTimer = 0;
	}*/



	switch(Cur_VSTATE)
	{

		case VSTATE_SyncWait:
		{
				//Waiting for SCDT on interrupt !!!
				//if(VideoCountingTimer==0)

				WaitingForSCDT();

#if 0
				if(TimeOutCheck(eVideoCountingTimer)==TRUE) {
					MHLRX_DEBUG_PRINTF(("------------SyncWait time out -----------\n"));
					SWReset_HDMIRX();
					return;
				}
#endif
		}
		break;

		case VSTATE_SyncChecking:
		{
			        //if(VideoTimeOutCheck(VSATE_CONFIRM_SCDT_COUNT))
			       // if(it6802->m_VideoCountingTimer == 0)//remark June
			bSynWaitcnt--;
			if ((bSynWaitEn == TRUE )&&(bSynWaitcnt ==0))
			{
				IT6801SwitchVideoState(VSTATE_VideoOn);
				bSynWaitEn = FALSE;
			}
		}
		break;

		case VSTATE_VideoOn:
		{
				if(m_NewAVIInfoFrameF==TRUE)
				{
					if(m_RxHDCPState != 0x01)//
					{
						IT6801VideoOutputConfigure();
						m_NewAVIInfoFrameF=FALSE;
					}

				}

				Read_Ite6801_i2c(ITE_HDMI_I2C_ADDR, REG_RX_53, &rxdata);
				rxdata &=0x80;
				//IT6801_SetVideoMute(FALSE);

				if(rxdata)
				{
					if(CheckAVMute()==FALSE)
			        {
						IT6801_SetVideoMute(FALSE);
			        }
				}

//				#ifdef Enable_Vendor_Specific_packet
//					if(it6802->f_de3dframe_hdmi == FALSE)
//					{
//					it6802->f_de3dframe_hdmi = IT6802_DE3DFrame(TRUE);
//					}
//				#endif

			}
			break;
	}
}

void ITE6801_polling_input(void)
{
	//IT6801MHLInterruptHandler();
	IT6801HDMIInterruptHandler();
	IT6801VideoHandler();



}

static uint08 Ite6801RegSet(uint08 slaveaddr, uint08  offset, uint08  mask, uint08  ucdata )
{
	uint08 readtemp;
	uint08 writetemp;
	uint08 staus;

	staus=Read_Ite6801_i2c(slaveaddr,offset,&readtemp);
	writetemp = (readtemp &(~mask))|(mask&ucdata);
	staus|=Write_Ite6801_i2c(slaveaddr,offset,&writetemp);
	if (staus!=0)
		return FALSE;
	else
		return TRUE;


	//temp = (temp&((~mask)&0xFF))+(mask&ucdata);
	//return hdmirxwr(offset, temp);
}

uint08 Write_Ite6801_i2c(uint08 slavaddr, uint08 offset, uint08* data)
{
	uint08 i2c_array[2];
	uint08 status;
	uint08 num_written;

	 i2c_array[0] = offset;
	 i2c_array[1] = *data;

	  status = i2c_master_polled_write(slavaddr, i2c_array,2, &num_written, 30);
	  //while (UCB0STAT & UCBBUSY){};

	  if (status!=0)
		  return FALSE;
	  else
		  return TRUE;

}

uint08 Read_Ite6801_i2c(uint08 slaveaddr, uint08 offset, uint08* data)
{
	uint08 num_written;
	uint08 status;
  	uint08 i2c_cmd;
  	uint08 bytes_read;

  	i2c_cmd = offset;

  	// write request
	status = i2c_master_polled_write(slaveaddr, &i2c_cmd, 1, &num_written, 30);
	//while (UCB0STAT & UCBBUSY){};

	if (status == 0)
	{
		status = i2c_master_polled_read(slaveaddr, data, 1, &bytes_read, 30);
		//while (UCB0STAT & UCBBUSY){};
	}

  if ( status != 0)
  	return FALSE;
  else
  	return TRUE;
}


