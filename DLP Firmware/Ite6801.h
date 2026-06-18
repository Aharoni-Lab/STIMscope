/*
 * Ite6801.h
 *
 *  Created on: 2014/12/3
 *      Author: june.liao
 */

#ifndef ITE6801_H_
#define ITE6801_H_

#define ITE_HDMI_I2C_ADDR 0x90
#define MHL_ADDR 0xE0

#define _SUPPORT_RCP_
#define _SUPPORT_RAP_

#define VSTATE_Off				0x00
#define VSTATE_TerminationOff	0x01
#define VSTATE_TerminationOn	0x02
#define VSTATE_5VOff			0x03
#define VSTATE_SyncWait			0x04
#define VSTATE_SWReset			0x05
#define VSTATE_SyncChecking		0x06
#define VSTATE_HDCPSet			0x07
#define VSTATE_HDCP_Reset		0x08
#define VSTATE_ModeDetecting	0x09
#define VSTATE_VideoOn			0x0A
#define VSTATE_ColorDetectReset	0x0B
#define VSTATE_HDMI_OFF			0x0C
#define VSTATE_Reserved			0x0D

#define F_MODE_RGB24  0
#define F_MODE_RGB444  0
#define F_MODE_YUV422 1
#define F_MODE_YUV444 2
#define F_MODE_CLRMOD_MASK 3
#define F_MODE_ITU709  (1<<4)
#define F_MODE_ITU601  0
#define F_MODE_0_255   0
#define F_MODE_16_235  (1<<5)
#define F_MODE_EN_UDFILT (1<<6)
#define F_MODE_EN_DITHER (1<<7)

#define M_CSC_SEL_MASK   0x03	
#define B_CSC_BYPASS        0x00
#define B_CSC_RGB2YUV      0x02	// for Andrew modify to 10
#define B_CSC_YUV2RGB      0x03


#define F_PORT_SEL_0      0

#define DeltaNum 	1
#define  RCLKFreqSel 	1	//; //0: RING/2 ; 1: RING/4 ; 2: RING/8 ; 3: RING/16
#define GenPktRecType	0x81
#define B_CTS_RES 0x70 // bit6~4
#define B_HBRSel 0x40 // bit6

#define B_DisVAutoMute    0x20

//#define REG_RX_OUPT_CTRL2  0x53 // REG_RX_053
	#define B_VDGatting 0x80 // bit7 -> Enable output data gating to zero when no Video display
	#define B_VIOSel 0x40 // bit6 -> 1: video IO enable depent on VIOenable
	#define B_VDIOLLdisable 0x20 // bit5 -> 1: disable video IO QE0, QE1, QE12, QE13, QE24, QE25
	#define B_VDIOLHdisable 0x10 // bit4 -> 1:  disable video IO QE2, QE3, QE14, QE15, QE26, QE27
	#define B_TriVDIO 0x0E // bit2~1 -> 111: enable tri-state Video IO
	#define B_TriSYNC 0x01 // bit0 ->1: Tristate video control signal IO

//#define REG_RX_P0_SYS_STATUS 0x0A
	#define B_P0_SCDT 0x80 // bit7
	#define B_P0_MHL_MODE 0x40 // bit6
	#define B_P0_IPLL_LOCK 0x20 // bit5
	#define B_P0_RXCK_SPEED 0x10 // bit4
	#define B_P0_RXCK_VALID 0x08 // bit3
	#define B_P0_VCLK_DET 0x04 // bit2
	#define B_P0_HDMI_MODE 0x02 // bit1
	#define B_P0_PWR5V_DET 0x01 // bit0

//#define REG_RX_051 0x51 // REG_RX_051
	#define B_PORT_SEL 0x01 // bit0
	#define B_EN_DEBUG 0x02 // bit1
	#define B_CCIR656 0x04 // bit2
	#define B_DisPixRpt 0x08 // bit3
	#define B_HALF_CLK 0x10 // bit4
	#define B_OUT_DDR 0x20 // bit5
	#define B_HALF_PCLKC 0x40 // bit6
	#define B_PWD_CSC 0x80 // bit7

//#define REG_RX_VIDEO_CTRL1 0x67 // REG_RX_067
	#define B_RX_EN_UDFILTER 0x01 // bit0
	#define B_RX_EN_DITHER 0x02 // bit1
	#define B_RX_DNFREE_GO 0x04 // bit2
	//#define B_3 0x08 // bit3
	//#define B_4 0x10 // bit4
	//#define B_5 0x20 // bit5
	//#define B_6 0x40 // bit6
	#define B_AutoCSCSel 0x80 // bit7

#define B_IN_FORCE_COLOR_MODE 0x04 // bit2
#define M_INPUT_COLOR_MASK 0x03
#define B_INPUT_RGB24      0x00
#define B_INPUT_YUV422     0x01
#define B_INPUT_YUV444     0x02
#define B_BTA1004Fmt 0x80 // bit7
#define B_SyncEmb 0x40 // bit6

#define B_OUTPUT_RGB24      0x00

#define DEFAULT_EQVALUE 0x1F


/*****************************************************************************/
/* Register Definitions ******************************************************/
/*****************************************************************************/


#define BANK0 0x00
#define BANK1 0x01
#define BANK2 0x02

#define REG_RX_00 0x00 
#define REG_RX_01 0x01
#define REG_RX_02 0x02 
#define REG_RX_03 0x03 
#define REG_RX_04 0x04 
#define REG_RX_05 0x05 
#define REG_RX_06 0x06 
#define REG_RX_07 0x07 
#define REG_RX_08 0x08 
#define REG_RX_09 0x09
#define REG_RX_0A 0x0A 
#define REG_RX_0B 0x0B 
#define REG_RX_0C 0x0C 
#define REG_RX_0D 0x0D 
#define REG_RX_0E 0x0E 
#define REG_RX_0F 0x0F 
#define REG_RX_10 0x10 
#define REG_RX_11 0x11 
#define REG_RX_12 0x12 
#define REG_RX_13 0x13 
#define REG_RX_14 0x14 
#define REG_RX_15 0x15 
#define REG_RX_16 0x16 
#define REG_RX_17 0x17 
#define REG_RX_18 0x18 
#define REG_RX_19 0x19 
#define REG_RX_1A 0x1A 
#define REG_RX_1B 0x1B 
#define REG_RX_1C 0x1C 
#define REG_RX_1D 0x1D 
#define REG_RX_1E 0x1E 
#define REG_RX_1F 0x1F 
#define REG_RX_20 0x20 
#define REG_RX_21 0x21 
#define REG_RX_22 0x22 
#define REG_RX_23 0x23 
#define REG_RX_24 0x24 
#define REG_RX_25 0x25 
#define REG_RX_26 0x26 
#define REG_RX_27 0x27 
#define REG_RX_28 0x28 
#define REG_RX_29 0x29 
#define REG_RX_2A 0x2A 
#define REG_RX_2B 0x2B 
#define REG_RX_2C 0x2C 
#define REG_RX_2D 0x2D 
#define REG_RX_2E 0x2E 
#define REG_RX_2F 0x2F 
#define REG_RX_30 0x30 
#define REG_RX_31 0x31 
#define REG_RX_32 0x32 
#define REG_RX_33 0x33 
#define REG_RX_34 0x34 
#define REG_RX_35 0x35 
#define REG_RX_36 0x36 
#define REG_RX_37 0x37 
#define REG_RX_38 0x38 
#define REG_RX_39 0x39 
#define REG_RX_3A 0x3A 
#define REG_RX_3B 0x3B 
#define REG_RX_3C 0x3C 
#define REG_RX_3D 0x3D 
#define REG_RX_3E 0x3E 
#define REG_RX_3F 0x3F 
#define REG_RX_40 0x40 
#define REG_RX_41 0x41 
#define REG_RX_42 0x42 
#define REG_RX_43 0x43 
#define REG_RX_44 0x44 
#define REG_RX_45 0x45 
#define REG_RX_46 0x46 
#define REG_RX_47 0x47 
#define REG_RX_48 0x48 
#define REG_RX_49 0x49 
#define REG_RX_4A 0x4A 
#define REG_RX_4B 0x4B 
#define REG_RX_4C 0x4C 
#define REG_RX_4D 0x4D 
#define REG_RX_4E 0x4E 
#define REG_RX_4F 0x4F 
#define REG_RX_50 0x50 
#define REG_RX_51 0x51 
#define REG_RX_52 0x52 
#define REG_RX_53 0x53 
#define REG_RX_54 0x54 
#define REG_RX_55 0x55 
#define REG_RX_56 0x56 
#define REG_RX_57 0x57 
#define REG_RX_58 0x58 
#define REG_RX_59 0x59 
#define REG_RX_5A 0x5A 
#define REG_RX_5B 0x5B 
#define REG_RX_5C 0x5C 
#define REG_RX_5D 0x5D 
#define REG_RX_5E 0x5E 
#define REG_RX_5F 0x5F
#define REG_RX_60 0x60
#define REG_RX_61 0x61
#define REG_RX_62 0x62
#define REG_RX_63 0x63
#define REG_RX_64 0x64
#define REG_RX_65 0x65
#define REG_RX_66 0x66
#define REG_RX_67 0x67
#define REG_RX_68 0x68
#define REG_RX_69 0x69
#define REG_RX_6A 0x6A
#define REG_RX_6B 0x6B
#define REG_RX_6C 0x6C
#define REG_RX_6D 0x6D
#define REG_RX_6E 0x6E
#define REG_RX_6F 0x6F
#define REG_RX_70 0x70
#define REG_RX_71 0x71
#define REG_RX_72 0x72
#define REG_RX_73 0x73
#define REG_RX_74 0x74
#define REG_RX_75 0x75
#define REG_RX_76 0x76
#define REG_RX_77 0x77
#define REG_RX_78 0x78
#define REG_RX_79 0x79
#define REG_RX_7A 0x7A
#define REG_RX_7B 0x7B
#define REG_RX_7C 0x7C
#define REG_RX_7D 0x7D
#define REG_RX_7E 0x7E
#define REG_RX_7F 0x7F
#define REG_RX_80 0x80
#define REG_RX_81 0x81
#define REG_RX_82 0x82
#define REG_RX_83 0x83
#define REG_RX_84 0x84
#define REG_RX_85 0x85
#define REG_RX_86 0x86
#define REG_RX_87 0x87
#define REG_RX_88 0x88
#define REG_RX_89 0x89
#define REG_RX_8A 0x8A
#define REG_RX_8B 0x8B
#define REG_RX_8C 0x8C
#define REG_RX_8D 0x8D
#define REG_RX_8E 0x8E
#define REG_RX_8F 0x8F
#define REG_RX_90 0x90
#define REG_RX_91 0x91
#define REG_RX_92 0x92
#define REG_RX_93 0x93
#define REG_RX_94 0x94
#define REG_RX_95 0x95
#define REG_RX_96 0x96
#define REG_RX_97 0x97
#define REG_RX_98 0x98
#define REG_RX_99 0x99
#define REG_RX_9A 0x9A
#define REG_RX_9B 0x9B
#define REG_RX_9C 0x9C
#define REG_RX_9D 0x9D 
#define REG_RX_9E 0x9E 
#define REG_RX_9F 0x9F 
#define REG_RX_A0 0xA0 
#define REG_RX_A1 0xA1 
#define REG_RX_A2 0xA2 
#define REG_RX_A3 0xA3 
#define REG_RX_A4 0xA4 
#define REG_RX_A5 0xA5 
#define REG_RX_A6 0xA6
#define REG_RX_A7 0xA7 
#define REG_RX_A8 0xA8 
#define REG_RX_A9 0xA9 
#define REG_RX_AA 0xAA 
#define REG_RX_AB 0xAB 
#define REG_RX_AC 0xAC 
#define REG_RX_AD 0xAD 
#define REG_RX_AE 0xAE 
#define REG_RX_AF 0xAF 
#define REG_RX_BA 0xBA 
#define REG_RX_BB 0xBB 
#define REG_RX_BC 0xBC 
#define REG_RX_BD 0xBD 
#define REG_RX_BE 0xBE 
#define REG_RX_BF 0xBF 
#define REG_RX_B0 0xB0 
#define REG_RX_B1 0xB1 
#define REG_RX_B2 0xB2 
#define REG_RX_B3 0xB3 
#define REG_RX_B4 0xB4 
#define REG_RX_B5 0xB5 
#define REG_RX_B6 0xB6 
#define REG_RX_B7 0xB7 
#define REG_RX_B8 0xB8 
#define REG_RX_B9 0xB9 
#define REG_RX_BA 0xBA 
#define REG_RX_BB 0xBB
#define REG_RX_BC 0xBC 
#define REG_RX_BD 0xBD 
#define REG_RX_BE 0xBE 
#define REG_RX_BF 0xBF 
#define REG_RX_BA 0xBA 
#define REG_RX_BB 0xBB 
#define REG_RX_BC 0xBC 
#define REG_RX_BD 0xBD 
#define REG_RX_BE 0xBE 
#define REG_RX_BF 0xBF 
#define REG_RX_C0 0xC0 
#define REG_RX_C1 0xC1 
#define REG_RX_C2 0xC2 
#define REG_RX_C3 0xC3 
#define REG_RX_C4 0xC4 
#define REG_RX_C5 0xC5 
#define REG_RX_C6 0xC6
#define REG_RX_C7 0xC7 
#define REG_RX_C8 0xC8 
#define REG_RX_C9 0xC9 
#define REG_RX_CA 0xCA
#define REG_RX_CB 0xCB 
#define REG_RX_CC 0xCC 
#define REG_RX_CD 0xCD 
#define REG_RX_CE 0xCE 
#define REG_RX_CF 0xCF 
#define REG_RX_CA 0xCA 
#define REG_RX_CB 0xCB 
#define REG_RX_CC 0xCC 
#define REG_RX_CD 0xCD 
#define REG_RX_CE 0xCE 
#define REG_RX_CF 0xCF 
#define REG_RX_D0 0xD0 
#define REG_RX_D1 0xD1
#define REG_RX_D2 0xD2 
#define REG_RX_D3 0xD3 
#define REG_RX_D4 0xD4 
#define REG_RX_D5 0xD5 
#define REG_RX_D6 0xD6 
#define REG_RX_D7 0xD7 
#define REG_RX_D8 0xD8 
#define REG_RX_D9 0xD9 
#define REG_RX_DA 0xDA 
#define REG_RX_DB 0xDB 
#define REG_RX_DC 0xDC 
#define REG_RX_DD 0xDD 
#define REG_RX_DE 0xDE 
#define REG_RX_DF 0xDF 
#define REG_RX_DA 0xDA 
#define REG_RX_DB 0xDB 
#define REG_RX_DC 0xDC 
#define REG_RX_DD 0xDD 
#define REG_RX_DE 0xDE 
#define REG_RX_DF 0xDF 
#define REG_RX_E0 0xE0 
#define REG_RX_E1 0xE1 
#define REG_RX_E2 0xE2
#define REG_RX_E3 0xE3
#define REG_RX_E4 0xE4
#define REG_RX_E5 0xE5
#define REG_RX_E6 0xE6 
#define REG_RX_E7 0xE7 
#define REG_RX_E8 0xE8 
#define REG_RX_E9 0xE9 
#define REG_RX_EA 0xEA 
#define REG_RX_EB 0xEB 
#define REG_RX_EC 0xEC 
#define REG_RX_ED 0xED 
#define REG_RX_EE 0xEE 
#define REG_RX_EF 0xEF 
#define REG_RX_EA 0xEA 
#define REG_RX_EB 0xEB 
#define REG_RX_EC 0xEC 
#define REG_RX_ED 0xED 
#define REG_RX_EE 0xEE 
#define REG_RX_EF 0xEF 
#define REG_RX_F0 0xF0 
#define REG_RX_F1 0xF1 
#define REG_RX_F2 0xF2 
#define REG_RX_F3 0xF3 
#define REG_RX_F4 0xF4 
#define REG_RX_F5 0xF5 
#define REG_RX_F6 0xF6 
#define REG_RX_F7 0xF7 
#define REG_RX_F8 0xF8 
#define REG_RX_F9 0xF9 
#define REG_RX_FA 0xFA 
#define REG_RX_FB 0xFB 
#define REG_RX_FC 0xFC 
#define REG_RX_FD 0xFD 
#define REG_RX_FE 0xFE 
#define REG_RX_FF 0xFF 
#define REG_RX_FA 0xFA 
#define REG_RX_FB 0xFB 
#define REG_RX_FC 0xFC 
#define REG_RX_FD 0xFD 
#define REG_RX_FE 0xFE 
#define REG_RX_FF 0xFF 


typedef struct IT6801_REG_INI
{
    unsigned char ucAddr;
    unsigned char andmask;
    unsigned char ucValue;
}IT6801_REG_INI;

void ITE6801_Init(void);
void ITE6801_polling_input(void);
uint08 GetITE6801CurStatus(void);
void GetPicoResolution(uint16 *HActive,uint16 *VActive );


#endif /* ITE6801_H_ */
