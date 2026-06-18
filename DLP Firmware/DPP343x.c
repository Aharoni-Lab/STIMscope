#include "common.h"
#include "DPP343x.h"
#include "i2c_master.h"

uint08 para[10];
uint08 para_length;

uint08 cmdoffset[256]=
{
	255, //0x00
	94, //0x01
	255, //0x02
	255, //0x03
	255, //0x04
	0,   //0x05
	1,   //0x06
	2,   //0x07
	3,   //0x08
	4,   //0x09
	5,   //0x0A
	6,   //0x0B
	7,   //0x0C
	8,   //0x0D
	9,  //0x0E
	10,  //0x0F
	11,  //0x10
	12,  //0x11
	13,  //0x12
	14,  //0x13
	15,  //0x14
	16,  //0x15
	17,  //0x16
	18,  //0x17
	255, //0x18
	255, //0x19
	19,  //0x1A
	20,  //0x1B
	255, //0x1C
	255, //0x1D
	255, //0x1E
	255, //0x1F
	21,  //0x20
	22,  //0x21
	23,  //0x22
	24,  //0x23
	255, //0x24
	255, //0x25
	25, //0x26
	26, //0x27
	27, //0x28
	28, //0x29
	29, //0x2A
	255, //0x2B
	255, //0x2C
	30, //0x2D
	31, //0x2E
	32, //0x2F
	33, //0x30
	34, //0x31
	35, //0x32
	36, //0x33
	37, //0x34
	38, //0x35
	39, //0x36
	40, //0x37
	41, //0x38
	255, //0x39
	255, //0x3A
	255, //0x3B
	255, //0x3C
	255, //0x3D
	255, //0x3E
	255, //0x3F
	255, //0x40
	255, //0x41
	255, //0x42
	255, //0x43
	255, //0x44
	255, //0x45
	255, //0x46
	255, //0x47
	255, //0x48
	255, //0x49
	255, //0x4A
	255, //0x4B
	255, //0x4C
	255, //0x4D
	255, //0x4E
	255, //0x4F
	42, //0x50
	43, //0x51
	44, //0x52
	45, //0x53
	46, //0x54
	47, //0x55
	255, //0x56
	48, //0x57
	255, //0x58
	255, //0x59
	255, //0x5A
	255, //0x5B
	49, //0x5C
	50, //0x5D
	51, //0x5E
	52, //0x5F
	255, //0x60
	255, //0x61
	255, //0x62
	255, //0x63
	255, //0x64
	255, //0x65
	255, //0x66
	255, //0x67
	255, //0x68
	255, //0x69
	255, //0x6A
	255, //0x6B
	255, //0x6C
	255, //0x6D
	255, //0x6E
	255, //0x6F
	255, //0x70
	255, //0x71
	255, //0x72
	255, //0x73
	255, //0x74
	255, //0x75
	255, //0x76
	255, //0x77
	255, //0x78
	255, //0x79
	255, //0x7A
	255, //0x7B
	255, //0x7C
	255, //0x7D
	255, //0x7E
	255, //0x7F
	53, //0x80
	54, //0x81
	255, //0x82
	255, //0x83
	55, //0x84
	56, //0x85
	57, //0x86
	58, //0x87
	255, //0x88
	255, //0x89
	255, //0x8A
	255, //0x8B
	255, //0x8C
	255, //0x8D
	255, //0x8E
	255, //0x8F
	255, //0x90
	255, //0x91
	91, //0x92
	255, //0x93
	255, //0x94
	255, //0x95
	92, //0x96
	255, //0x97
	255, //0x98
	255, //0x99
	255, //0x9A
	255, //0x9B
	255, //0x9C
	93, //0x9D
	255, //0x9E
	255, //0x9F
	255, //0xA0
	255, //0xA1
	255, //0xA2
	255, //0xA3
	255, //0xA4
	255, //0xA5
	255, //0xA6
	255, //0xA7
	255, //0xA8
	255, //0xA9
	255, //0xAA
	255, //0xAB
	255, //0xAC
	255, //0xAD
	255, //0xAE
	255, //0xAF
	255, //0xB0
	255, //0xB1
	59, //0xB2
	60, //0xB3
	255, //0xB4
	255, //0xB5
	61, //0xB6
	62, //0xB7
	63, //0xB8
	64, //0xB9
	65, //0xBA
	255, //0xBB
	255, //0xBC
	255, //0xBD
	255, //0xBE
	255, //0xBF
	255, //0xC0
	255, //0xC1
	255, //0xC2
	255, //0xC3
	255, //0xC4
	255, //0xC5
	255, //0xC6
	255, //0xC7
	255, //0xC8
	255, //0xC9
	255, //0xCA
	255, //0xCB
	255, //0xCC
	255, //0xCD
	255, //0xCE
	255, //0xCF
	66, //0xD0
	67, //0xD1
	68, //0xD2
	69, //0xD3
	70, //0xD4
	71, //0xD5
	72, //0xD6
	255, //0xD7
	255, //0xD8
	73, //0xD9
	255, //0xDA
	74, //0xDB
	75, //0xDC
	76, //0xDD
	77, //0xDE
	78, //0xDF
	79, //0xE0
	80, //0xE1
	81, //0xE2
	82, //0xE3
	83, //0xE4
	84, //0xE5
	85, //0xE6
	86, //0xE7
	87, //0xE8
	88, //0xE9
	89, //0xEA
	90, //0xEB
	95, //0xEC
	96, //0xED
	255, //0xEE
	255, //0xEF
	255, //0xF0
	255, //0xF1
	255, //0xF2
	255, //0xF3
	255, //0xF4
	255, //0xF5
	255, //0xF6
	255, //0xF7
	255, //0xF8
	255, //0xF9
	255, //0xFA
	255, //0xFB
	255, //0xFC
	255, //0xFD
	255, //0xFE
	255, //0xFF
};

static DDP343XCMD dpp3439cmdlist[]=
{
	{ W_INPUT_SOURCE_SELECT,0,1,0},					// 0x05//0
	{ R_INPUT_SOURCE_SELECT,1,1,1},					// 0x06//1
	{ W_EXT_VIDEO_FORMAT_SELECT,0,1,0}, 			// 0x07//2
	{ R_EXT_VIDEO_FORMAT_SELECT,1,1,1}, 			// 0x08//3
	{ W_EXT_VIDEO_CHROMA_PROCESSING_SELECT,0,2,0},	// 0x09//4
	{ R_EXT_VIDEO_CHROMA_PROCESSING_SELECT,1,1,2},  // 0x0A//5
	{ W_TEST_PATTERN_SELECT,0,255,0}, 				// 0x0B//6
	{ R_TEST_PATTERN_SELECT,1,1,6}, 				// 0x0C//7
	{ W_SPLASH_SCREEN_SELECT,0,1,0}, 				// 0x0D//8
	{ R_SPLASH_SCREEN_SELECT,1,1,1},				// 0x0E//9
	{ R_SPLASH_SCREEN_HEADER,1,2,13}, 				// 0x0F//10
	{ W_IMAGE_CROP,0,8,0}, 							// 0x10//11
	{ R_IMAGE_CROP,1,1,8}, 							// 0x11//12
	{ W_DISPLAY_SIZE,0,4,0}, 						// 0x12//13
	{ R_DISPLAY_SIZE,1,1,4}, 						// 0x13//14
	{ W_DISPLAY_IMAGE_ORIENTATION,0,1,0}, 			// 0x14//15
	{ R_DISPLAY_IMAGE_ORIENTATION,1,1,1}, 			// 0x15//16
	{ W_DISPLAY_IMAGE_CURTAIN,0,1,0}, 				// 0x16//17
	{ R_DISPLAY_IMAGE_CURTAIN,1,1,1}, 				// 0x17//18
	{ W_IMAGE_FREEZE,0,1,0}, 						// 0x1A//19
	{ R_IMAGE_FREEZE,1,1,1}, 						// 0x1B//20
	{ W_3D_CTRL,0,1,0}, 							// 0x20//21
	{ R_3D_CTRL,1,1,1}, 							// 0x21//22
	{ W_LOOK_SELECT,0,1,0}, 						// 0x22//23
	{ R_LOOK_SELECT,1,1,6}, 						// 0x23//24
	{ R_SEQUENCE_HEADER_ATTRIBUTES,1,1,30}, 		// 0x26//25
	{ W_DEGAMMA_CMT_SELECT,0,1,0}, 					// 0x27//26
	{ R_DEGAMMA_CMT_SELECT,1,1,1}, 					// 0x28//27
	{ W_CCA_SELECT,0,1,0}, 							// 0x29
	{ R_CCA_SELECT,1,1,1}, 							// 0x2A
	{ W_EXECUTE_BATCH_FILE,0,1,0}, 					// 0x2D
	{ W_EXTERNAL_INPUT_IMAGE_SIZE,0,4,0}, 			// 0x2E
	{ R_EXTERNAL_INPUT_IMAGE_SIZE,1,1,4},			// 0x2F
	{ W_3D_REFERENCE,0,1,0},						// 0x30
	{ W_GPIO_19_00_CTRL,0,4,0}, 					// 0x31
	{ R_GPIO_19_00_CTRL,1,1,4}, 					// 0x32
	{ W_GPIO_19_00_OUTPUT,0,6,0}, 					// 0x33
	{ R_GPIO_19_00_OUTPUT,1,1,3}, 					// 0x34
	{ W_SPLASH_SCREEN_EXECUTE,0,0,0}, 				// 0x35
	{ R_GPIO_19_00_INPUT,1,1,3}, 					// 0x36
	{ W_EXTERNAL_PARALLEL_I_F_DATA_MASK_CONTROL,0,1,0}, // 0x37
	{ R_EXTERNAL_PARALLEL_I_F_DATA_MASK_CONTROL,1,1,1}, // 0x38
	{ W_LED_OUTPUT_CTRL_MOTHOD,0,1,0}, 				// 0x50
	{ R_LED_OUTPUT_CTRL_MOTHOD,1,1,1}, 				// 0x51
	{ W_RGB_LED_ENABLE,0,1,0}, 						// 0x52
	{ R_RGB_LED_ENABLE,1,1,1}, 						// 0x53
	{ W_MANUAL_RGB_LED_CURRENT,0,6,0}, 				// 0x54
	{ R_MANUAL_RGB_LED_CURRENT,1,1,6}, 				// 0x55
	{ R_CAIC_LED_MAX_AVAILABLE_POWER,1,1,2}, 		// 0x57
	{ W_MANUAL_RGB_LED_MAX_CURRENT,0,6,0}, 			// 0x5C
	{ R_MANUAL_RGB_LED_MAX_CURRENT,1,1,6}, 			// 0x5D
	{ R_MEASURED_LED_PARAMETERS,1,1,20}, 			// 0x5E
	{ R_CAIC_RGB_LED_CURRENT,1,1,6}, 				// 0x5F
	{ W_LOCAL_AREA_BRIGHTNESS_BOOST_CTRL,0,2,0}, 	// 0x80
	{ R_LOCAL_AREA_BRIGHTNESS_BOOST_CTRL,1,1,3}, 	// 0x81
	{ W_CAIC_IMAGE_PROCESSING_CTRL,0,3,0}, 			// 0x84
	{ R_CAIC_IMAGE_PROCESSING_CTRL,1,1,3}, 			// 0x85
	{ W_CCA_CTRL,0,1,0}, 							// 0x86
	{ R_CCA_CTRL,1,1,1}, 							// 0x87
	{ W_BORDER_COLOR,0,1,0}, 						// 0xB2
	{ R_BORDER_COLOR,1,1,1}, 						// 0xB3
	{ W_EXT_PARALLEL_I_F_SYNC_POLARITY,0,1,0}, 		// 0xB6
	{ R_EXT_PARALLEL_I_F_SYNC_POLARITY,1,1,1}, 		// 0xB7
	{ W_EXT_PARALLEL_I_F_MAUNAL_IMAGE_FRAMING,0,5,0}, // 0xB8
	{ R_EXT_PARALLEL_I_F_MAUNAL_IMAGE_FRAMING,1,1,5}, // 0xB9
	{ R_AUTO_FRAMING_INFO,1,1,14}, 					// 0xBA
	{ R_SHORT_STATUS,1,1,1}, 						// 0xD0
	{ R_SYSTEM_STATUS,1,1,4}, 						// 0xD1
	{ R_SYSTEM_SOFTWARE_VERSION,1,1,4}, 			// 0xD2
	{ R_COMMUNICATION_STATUS,1,2,6}, 				// 0xD3
	{ R_ASIC_DEVICE_ID,1,1,1}, 						// 0xD4
	{ R_DMD_DEVICE_ID,1,1,1}, 						// 0xD5
	{ R_SYSTEM_TEMPERATURE,1,1,2}, 					// 0xD6
	{ R_FLASH_VERSION,1,1,4}, 						// 0xD9
	{ W_BATCH_FILE_DELAY,0,2,0}, 					// 0xDB
	{ R_DMD_I_F_TRAINING_DATA,1,2,11}, 				// 0xDC
	{ R_FLASH_UPDATE_PRECHECK,1,4,1}, 				// 0xDD
	{ W_FLASH_DATA_TYPE_SELECT,0,4,0}, 				// 0xDE
	{ W_FLASH_DATA_LENGTH,0,2,0}, 					// 0xDF
	{ W_ERASE_FLASH_DATA,0,4,0}, 					// 0xE0
	{ W_FLASH_START,0,255,0}, 						// 0xE1
	{ W_FLASH_CONTINUE,0,255,0}, 					// 0xE2
	{ R_FLASH_START,1,1,255}, 						// 0xE3
	{ R_FLASH_CONTINUE,1,1,255}, 					// 0xE4
	{ W_INTERNAL_REGISTER_ADDRESS,0,4,0}, 			// 0xE5
	{ W_INTERNAL_REGISTER,0,4,0}, 					// 0xE6
	{ R_INTERNAL_REGISTER,1,1,4}, 					// 0xE7
	{ W_INTERNAL_MAILBOX_ADDRESS,0,17,0}, 			// 0xE8
	{ W_INTERNAL_MAILBOX,0,255,0}, 					// 0xE9
	{ R_INTERNAL_MAILBOX,1,1,255}, 					// 0xEA
	{ W_EXT_PAD_ADDRESS,0,5,0}, 					// 0xEB
	{ W_TRIG_OUT_CONFIG,0,5,0},                     // 0x92
	{ W_PATTERN_CONFIG,0,15,0},                     // 0x96
	{ W_READ_VALIDAT_EXPO_TIME,0,6,0},              // 0x9D
	{ R_H01DATA,1,1,14},                           // 0x01
	{ W_EXT_PAD_DATA,0,255,0}, 						// 0xEC
	{ R_EXT_PAD_DATA,1,1,255} 						// 0xED

};

/*static RCMD readcmd[]=
{
		{R_INPUT_SOURCE_SELECT,1},
		{R_EXT_VIDEO_FORMAT_SELECT,1},
		{R_EXT_VIDEO_CHROMA_PROCESSING_SELECT,2},
		{R_TEST_PATTERN_SELECT,6},
		{R_SPLASH_SCREEN_SELECT,1},
		{R_SPLASH_SCREEN_HEADER,13},
		{R_IMAGE_CROP,8},
		{R_DISPLAY_SIZE,4},
		{R_DISPLAY_IMAGE_ORIENTATION,1},
		{R_DISPLAY_IMAGE_CURTAIN,1},
		{R_IMAGE_FREEZE,1},
		{R_3D_CTRL,1},
		{R_LOOK_SELECT,6},
		{R_SEQUENCE_HEADER_ATTRIBUTES,30},
		{R_DEGAMMA_CMT_SELECT,1},
		{R_CCA_SELECT,1},
		{R_EXTERNAL_INPUT_IMAGE_SIZE,4},
		{R_GPIO_19_00_CTRL,4},
		{R_GPIO_19_00_OUTPUT,3},
		{R_GPIO_19_00_INPUT,3},
		{R_EXTERNAL_PARALLEL_I_F_DATA_MASK_CONTROL,1},
		{R_LED_OUTPUT_CTRL_MOTHOD,1},
		{R_RGB_LED_ENABLE,1},
		{R_MANUAL_RGB_LED_CURRENT,6},
		{R_CAIC_LED_MAX_AVAILABLE_POWER,2},
		{R_MANUAL_RGB_LED_MAX_CURRENT,6},
		{R_MEASURED_LED_PARAMETERS,20},
		{R_CAIC_RGB_LED_CURRENT,6},
		{R_LOCAL_AREA_BRIGHTNESS_BOOST_CTRL,3},
		{R_CAIC_IMAGE_PROCESSING_CTRL,3},
		{R_CCA_CTRL,1},
		{R_BORDER_COLOR,1},
		{R_EXT_PARALLEL_I_F_SYNC_POLARITY,1},
		{R_EXT_PARALLEL_I_F_MAUNAL_IMAGE_FRAMING,5},
		{R_AUTO_FRAMING_INFO,14},
		{R_SHORT_STATUS,1},
		{R_SYSTEM_STATUS,4},
		{R_SYSTEM_SOFTWARE_VERSION,4},
		{R_COMMUNICATION_STATUS,6},
		{R_ASIC_DEVICE_ID,1},
		{R_DMD_DEVICE_ID,4},
		{R_SYSTEM_TEMPERATURE,2},
		{R_FLASH_VERSION,4},
		{R_DMD_I_F_TRAINING_DATA,11},
		{R_EXT_PAD_DATA,1},
		{0xff,0x00}
};*/

extern void GetPicoResolution(uint16 *HActive,uint16 *VActive );

//uint08 write_dpp343x_i2c(uint08 addr, uint08 subaddr, uint08* data, uint08 length);
//uint08 Read_dpp343x_i2c(uint08 addr, uint08 subaddr, uint08* data, uint08 length);

void dpp343x_config_TPG(uint08 PatternSelect)
{
	
	// Image Crop(1920 x 1080)
	/*para[0] = 0x00;
	para[1] = 0x00;
	para[2] = 0x00;
	para[3] = 0x00;
	para[4] = 0x80;
	para[5] = 0x07;
	para[6] = 0x38;
	para[7] = 0x04;
	para_length = 8;
	write_dpp343x_i2c(DPP3438_DEV_ADDR, W_IMAGE_CROP, para, para_length);

	para[0] = 0x80;
	para[1] = 0x07;
	para[2] = 0x38;
	para[3] = 0x04;
	para_length = 4;
	write_dpp343x_i2c(DPP3438_DEV_ADDR, W_EXTERNAL_INPUT_IMAGE_SIZE, para, para_length);

	para[0] = 0x01;
	para_length = 1;
	write_dpp343x_i2c(DPP3438_DEV_ADDR, W_INPUT_SOURCE_SELECT, para, para_length);*/
	switch(PatternSelect)
	{
		case PAT_SOILD_FIELD_WHITE:
			para[0] = 0x00;
			para[1] = 0x70;
			para_length = 2;
		break;
		case PAT_SOLID_FILED_RED:
			para[0] = 0x00;
			para[1] = 0x10;
			para_length = 2;
		break;
		case PAT_SOILD_FIELD_GREEN:
			para[0] = 0x00;
			para[1] = 0x20;
			para_length = 2;
		break;
		case PAT_SOLID_FILED_BLUE:
			para[0] = 0x00;
			para[1] = 0x30;
			para_length = 2;
		break;
		case PAT_SOILD_FILED_BLACK:
			para[0] = 0x00;
			para[1] = 0x00;
			para_length = 2;
		break;
		case PAT_CHECKBOARD_5X5:
			para[0] = 0x07;
			para[1] = 0x70;
			para[2] = 0x05;
			para[3] = 0x00;
			para[4] = 0x05;
			para[5] = 0x00;
			para_length = 6;
		break;
		case PAT_CHECKBOARD_6X6:
			para[0] = 0x07;
			para[1] = 0x70;
			para[2] = 0x06;
			para[3] = 0x00;
			para[4] = 0x06;
			para[5] = 0x00;
			para_length = 6;
		break;
		case PAT_CHECKBOARD_32X18:
			para[0] = 0x07;
			para[1] = 0x70;
			para[2] = 0x20;
			para[3] = 0x00;
			para[4] = 0x12;
			para[5] = 0x00;
			para_length = 6;
		break;
		case PAT_CHECKBOARD_128X72:
			para[0] = 0x07;
			para[1] = 0x70;
			para[2] = 0x80;
			para[3] = 0x00;
			para[4] = 0x48;
			para[5] = 0x00;
			para_length = 6;
		break;
		default:
		break;
	}
	write_dpp343x_i2c(DPP343X_DEV_ADDR, W_TEST_PATTERN_SELECT, para, para_length);
	//para[0] = 0x01;
	//para_length = 1;
	//write_dpp343x_i2c(DPP3438_DEV_ADDR, W_INPUT_SOURCE_SELECT, para, para_length);
	
}

void dpp343x_source_input_select(uint08 source_select)
{
	uint16 HActive,VActive;
	switch (source_select)
	{
		case INPUT_EXTERNAL_HDMI:
			GetPicoResolution(&HActive,&VActive);
			//Extern_Source_Enable(TRUE);
			// Display Image Curtain (on/black)
			para[0] = 0x01;
			para_length = 1;
			write_dpp343x_i2c(DPP343X_DEV_ADDR, W_DISPLAY_IMAGE_CURTAIN, para, para_length);

			// Image Freeze
			para[0] = 0x01;
			para_length = 1;
			write_dpp343x_i2c(DPP343X_DEV_ADDR, W_IMAGE_FREEZE, para, para_length);

			// Image Crop(1280 x 720)
			para[0] = 0x00;
			para[1] = 0x00;
			para[2] = 0x00;
			para[3] = 0x00;
			para[4] = HActive%256;//0x80;
			para[5] = HActive/256;//0x07;
			para[6] = VActive%256;//0x38;
			para[7] = VActive/256;//0x04;
			para_length = 8;
			write_dpp343x_i2c(DPP343X_DEV_ADDR, W_IMAGE_CROP, para, para_length);

			// Display Size(1280 x 720)
			para[0] = 0x80;
			para[1] = 0x07;
			para[2] = 0x38;
			para[3] = 0x04;
			para_length = 4;
			write_dpp343x_i2c(DPP343X_DEV_ADDR, W_DISPLAY_SIZE, para, para_length);

			//External Input Image Size
			para[0] = HActive%256;//0x80;
			para[1] = HActive/256;//0x07;
			para[2] = VActive%256;//0x38;
			para[3] = VActive/256;//0x04;
			para_length = 4;
			write_dpp343x_i2c(DPP343X_DEV_ADDR, W_EXTERNAL_INPUT_IMAGE_SIZE, para, para_length);

			//External Video Source Format Select(Parallel / 24 / RGB 888)
			para[0] = 0x43;
			para_length = 1;
			write_dpp343x_i2c(DPP343X_DEV_ADDR, W_EXT_VIDEO_FORMAT_SELECT, para, para_length);

			//Input Source Select(External Video Port)
			para[0] = 0x00;
			para_length = 1;
			write_dpp343x_i2c(DPP343X_DEV_ADDR, W_INPUT_SOURCE_SELECT, para, para_length);

			// Image Unfreeze
			para[0] = 0x00;
			para_length = 1;
			write_dpp343x_i2c(DPP343X_DEV_ADDR, W_IMAGE_FREEZE, para, para_length);

			// Display Image Curtain (off/black)
			para[0] = 0x00;
			para_length = 1;
			write_dpp343x_i2c(DPP343X_DEV_ADDR, W_DISPLAY_IMAGE_CURTAIN, para, para_length);

		break;
		case INPUT_TEST_PATTERN:
			//Extern_Source_Enable(FALSE);
			para[0] = 0x01;
			para_length = 1;
			write_dpp343x_i2c(DPP343X_DEV_ADDR, W_INPUT_SOURCE_SELECT, para, para_length);
		break;
		case INPUT_SPLASH:
			// Image Crop(1920 x 1080)
			para[0] = 0x00;
			para[1] = 0x00;
			para[2] = 0x00;
			para[3] = 0x00;
			para[4] = 0x00;
			para[5] = 0x05;
			para[6] = 0xd0;
			para[7] = 0x02;
			/*para[4] = 0x80;
			para[5] = 0x07;
			para[6] = 0x38;
			para[7] = 0x04;*/
			para_length = 8;
			write_dpp343x_i2c(DPP343X_DEV_ADDR, W_IMAGE_CROP, para, para_length);

			// Display Size(1920 x 1080)
			para[0] = 0x80;
			para[1] = 0x07;
			para[2] = 0x38;
			para[3] = 0x04;
			para_length = 4;
			write_dpp343x_i2c(DPP343X_DEV_ADDR, W_DISPLAY_SIZE, para, para_length);

			//External Input Image Size
			/*para[0] = 0x80;
			para[1] = 0x07;
			para[2] = 0x38;
			para[3] = 0x04;*/
			para[0] = 0x00;
			para[1] = 0x05;
			para[2] = 0xd0;
			para[3] = 0x02;
			para_length = 4;
			write_dpp343x_i2c(DPP343X_DEV_ADDR, W_EXTERNAL_INPUT_IMAGE_SIZE, para, para_length);

			para[0] = 0x00;
			para_length = 1;
			write_dpp343x_i2c(DPP343X_DEV_ADDR, W_SPLASH_SCREEN_SELECT, para, para_length);

			para[0] = 0x02;
			para_length = 1;
			write_dpp343x_i2c(DPP343X_DEV_ADDR, W_INPUT_SOURCE_SELECT, para, para_length);

			para[0] = 0x00;
			para_length = 1;
			write_dpp343x_i2c(DPP343X_DEV_ADDR, W_SPLASH_SCREEN_EXECUTE, para, para_length);
			__delay_cycles(240000);
			__delay_cycles(240000);
		break;
		default:
		break;

	}
}

void* GetCmdInfo(uint08* pcamd)
{
	uint08 offset = 0;
	uint08 cmdid = 0;
	cmdid = *pcamd;
	if (cmdid != 0xFF)
	{
		offset = cmdoffset[cmdid];
		return (void*)(&dpp3439cmdlist[offset]);
	}
	else
		return NULL;
}

void SetWritePADDataLen(uint08 len)
{
	uint08 offset;
	offset = cmdoffset[W_EXT_PAD_DATA];
	dpp3439cmdlist[offset].wxlen = len;

}

void SetReadPADDataLen(uint08 len)
{
	uint08 offset;
	offset = cmdoffset[R_EXT_PAD_DATA];
	dpp3439cmdlist[offset].rxlen = len;

}

/*uint08 GetReadCmdlenght(uint08* pcmd)
{
	uint16 i;
	uint08 cmd;
	i=0;
	cmd = *pcmd;
	while (readcmd[i].cmdid != 0xff)
	{
		if (readcmd[i].cmdid == cmd)
			break;
		i++;
	}

	return readcmd[i].cmdlenght;
}*/




uint08 write_dpp343x_i2c(uint08 addr, uint08 subaddr, uint08* data, uint08 length)
{
  uint08 num_written;
  uint08 status;
  uint08 i2c_array[20];
  uint08 i;

  i2c_array[0] = subaddr;
  for (i = 0; i < length; i++)
  	i2c_array[i+1] = *data++;

  status = i2c_master_polled_write(addr, i2c_array, length+1, &num_written, 30);

  if ( status != 0)
  	return FALSE;
  else
  	return TRUE;

}

uint08 Read_dpp343x_i2c(uint08 addr, uint08* subaddr,uint08 txlen, uint08* data, uint08 length)
{
	uint08 num_written;
	uint08 status;
  	uint08 bytes_read;

  	// write request
	status = i2c_master_polled_write(addr, subaddr, txlen, &num_written, 30);

	if (status == 0)
	{
		status = i2c_master_polled_read(addr, data, length, &bytes_read, 30);
	}

  if ( status != 0)
  	return FALSE;
  else
  	return TRUE;
}




