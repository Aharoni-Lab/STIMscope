#ifndef DPP343x_H_
#define DPP343x_H_

#define DPP343X_DEV_ADDR 0x36

enum
{
	PAT_SOILD_FIELD_WHITE,
	PAT_SOLID_FILED_RED,
	PAT_SOILD_FIELD_GREEN,
	PAT_SOLID_FILED_BLUE,
	PAT_SOILD_FILED_BLACK,
	PAT_CHECKBOARD_5X5,
	PAT_CHECKBOARD_6X6,
	PAT_CHECKBOARD_32X18,
	PAT_CHECKBOARD_128X72,

};

enum
{
	INPUT_EXTERNAL_HDMI,
	INPUT_TEST_PATTERN,
	INPUT_SPLASH


};

/*typedef struct _RCMD
{
	uint08 cmdid;
	uint08 cmdlenght;
}RCMD;*/

typedef struct _DDP343XCMD
{
	uint08 id;
	uint08 type;
	uint08 wxlen;
	uint08 rxlen;
}DDP343XCMD;



void dpp343x_config_TPG(uint08 PatternSelect);
void dpp343x_source_input_select(uint08 source_select);
//uint08 GetReadCmdlenght(uint08* cmd);
void* GetCmdInfo(uint08* pcamd);
void SetWritePADDataLen(uint08 len);
void SetReadPADDataLen(uint08 len);
uint08 write_dpp343x_i2c(uint08 addr, uint08 subaddr, uint08* data, uint08 length);
//uint08 Read_dpp343x_i2c(uint08 addr, uint08 subaddr, uint08* data, uint08 length);
uint08 Read_dpp343x_i2c(uint08 addr, uint08* subaddr,uint08 txlen, uint08* data, uint08 length);


//General operation
#define W_INPUT_SOURCE_SELECT					0x05
#define R_INPUT_SOURCE_SELECT					0x06
#define W_EXT_VIDEO_FORMAT_SELECT				0x07
#define R_EXT_VIDEO_FORMAT_SELECT				0x08
#define W_EXT_VIDEO_CHROMA_PROCESSING_SELECT	0x09
#define R_EXT_VIDEO_CHROMA_PROCESSING_SELECT	0x0A
#define W_TEST_PATTERN_SELECT					0x0B
#define R_TEST_PATTERN_SELECT					0x0C
#define W_SPLASH_SCREEN_SELECT					0x0D
#define R_SPLASH_SCREEN_SELECT					0x0E
#define R_SPLASH_SCREEN_HEADER					0x0F
#define W_IMAGE_CROP							0x10
#define R_IMAGE_CROP							0x11
#define W_DISPLAY_SIZE							0x12
#define R_DISPLAY_SIZE							0x13
#define W_DISPLAY_IMAGE_ORIENTATION				0x14
#define R_DISPLAY_IMAGE_ORIENTATION				0x15
#define W_DISPLAY_IMAGE_CURTAIN					0x16
#define R_DISPLAY_IMAGE_CURTAIN					0x17
#define W_IMAGE_FREEZE							0x1A
#define R_IMAGE_FREEZE							0x1B

#define W_3D_CTRL								0x20
#define R_3D_CTRL								0x21
#define W_LOOK_SELECT							0x22
#define R_LOOK_SELECT							0x23
//#define W_MANUAL_DUTY_CYCLE_SELECT				0x24
//#define R_MANUAL_DUTY_CYCLE_SELECT				0x25
#define R_SEQUENCE_HEADER_ATTRIBUTES			0x26
#define W_DEGAMMA_CMT_SELECT					0x27
#define R_DEGAMMA_CMT_SELECT					0x28
#define W_CCA_SELECT							0x29
#define R_CCA_SELECT							0x2A
//#define W_DMD_SEQUENCER_SYNC_MODE				0x2B
//#define R_DMD_SEQUENCER_SYNC_MODE				0x2C
#define W_EXECUTE_BATCH_FILE					0x2D
#define W_EXTERNAL_INPUT_IMAGE_SIZE				0x2E
#define R_EXTERNAL_INPUT_IMAGE_SIZE				0x2F
#define W_3D_REFERENCE							0x30
#define W_GPIO_19_00_CTRL						0x31
#define R_GPIO_19_00_CTRL						0x32
#define W_GPIO_19_00_OUTPUT						0x33
#define R_GPIO_19_00_OUTPUT						0x34
#define W_SPLASH_SCREEN_EXECUTE					0x35
#define R_GPIO_19_00_INPUT						0x36
#define W_EXTERNAL_PARALLEL_I_F_DATA_MASK_CONTROL	0x37
#define R_EXTERNAL_PARALLEL_I_F_DATA_MASK_CONTROL	0x38
//#define R_DSI_FLASHLESS_DATA_QUERY				0x40
//#define W_DSI_FLASHLESS_DATA_LENGTH				0x41
//#define W_DSI_FLASHLESS_DATA_RESPONSE			0x42
//#define W_DSI_READ_PREFETCH						0x4E
//#define R_DSI_READ_ACTIVATE						0x4F

//Illumination Control
#define W_LED_OUTPUT_CTRL_MOTHOD				0x50
#define R_LED_OUTPUT_CTRL_MOTHOD				0x51
#define W_RGB_LED_ENABLE						0x52
#define R_RGB_LED_ENABLE						0x53
#define W_MANUAL_RGB_LED_CURRENT				0x54
#define R_MANUAL_RGB_LED_CURRENT				0x55
#define R_CAIC_LED_MAX_AVAILABLE_POWER			0x57
#define W_MANUAL_RGB_LED_MAX_CURRENT			0x5C
#define R_MANUAL_RGB_LED_MAX_CURRENT			0x5D
#define R_MEASURED_LED_PARAMETERS				0x5E
#define R_CAIC_RGB_LED_CURRENT					0x5F
//My illumination
#define W_TRIG_OUT_CONFIG                       0x92
#define W_PATTERN_CONFIG                        0x96
#define W_READ_VALIDAT_EXPO_TIME                0x9D
#define R_H01DATA                               0x01

//Image Processing Control
#define W_LOCAL_AREA_BRIGHTNESS_BOOST_CTRL		0x80
#define R_LOCAL_AREA_BRIGHTNESS_BOOST_CTRL		0x81
//#define W_SHALLOW_GRADIENT_PROCESSING_CTRL		0x82
//#define R_SHALLOW_GRADIENT_PROCESSING_CTRL		0x83
#define W_CAIC_IMAGE_PROCESSING_CTRL			0x84
#define R_CAIC_IMAGE_PROCESSING_CTRL			0x85
#define W_CCA_CTRL								0x86
#define R_CCA_CTRL								0x87
//#define W_KEYSTONE_CORRECTION_CTRL				0x88
//#define R_KEYSTONE_CORRECTION_CTRL				0x89
//#define W_TEMPORAL_DITHERING_CTRL				0x8A
//#define R_TEMPORAL_DITHERING_CTRL				0x8B
//#define W_BOUNDARY_DISPERSION_CTRL				0x8C
//#define R_BOUNDARY_DISPERSION_CTRL				0x8D
//#define W_FORCE_COAST_CTRL						0x8E
//#define R_FORCE_COAST_CTRL						0x8F

// General Setup
//#define W_INTERNAL_VSYNC_RATE					0xB0
//#define R_INTERNAL_VSYNC_RATE					0xB1
#define W_BORDER_COLOR							0xB2
#define R_BORDER_COLOR							0xB3
//#define W_EXT_CPU_I_F_VIDEO_SYNC_METHOD			0xB4
//#define R_EXT_CPU_I_F_VIDEO_SYNC_METHOD			0xB5
#define W_EXT_PARALLEL_I_F_SYNC_POLARITY		0xB6
#define R_EXT_PARALLEL_I_F_SYNC_POLARITY		0xB7
#define W_EXT_PARALLEL_I_F_MAUNAL_IMAGE_FRAMING	0xB8
#define R_EXT_PARALLEL_I_F_MAUNAL_IMAGE_FRAMING	0xB9
#define R_AUTO_FRAMING_INFO						0xBA
//#define W_KEYSTONE_PROJECTION_PICTCH_ANGLE		0xBB
//#define R_KEYSTONE_PROJECTION_PICTCH_ANGLE		0xBC
//#define W_DSI_PARAMETERS						0xBD
//#define R_DSI_PARAMETERS						0xBE

//Administrative Commands
#define R_SHORT_STATUS							0xD0
#define R_SYSTEM_STATUS							0xD1
#define R_SYSTEM_SOFTWARE_VERSION				0xD2
#define R_COMMUNICATION_STATUS					0xD3
#define R_ASIC_DEVICE_ID						0xD4
#define R_DMD_DEVICE_ID							0xD5
#define R_SYSTEM_TEMPERATURE					0xD6
//#define W_DSI_PORT_ENABLE						0xD7
//#define R_DSI_PORT_ENABLE						0xD8
#define R_FLASH_VERSION							0xD9
//#define R_RGB_LIGHT_SENSOR_DATA					0xDA
#define W_BATCH_FILE_DELAY						0xDB
#define R_DMD_I_F_TRAINING_DATA					0xDC
#define R_FLASH_UPDATE_PRECHECK					0xDD
#define W_FLASH_DATA_TYPE_SELECT				0xDE
#define W_FLASH_DATA_LENGTH						0xDF
#define W_ERASE_FLASH_DATA						0xE0
#define W_FLASH_START							0xE1
#define W_FLASH_CONTINUE						0xE2
#define R_FLASH_START							0xE3
#define R_FLASH_CONTINUE						0xE4
#define W_INTERNAL_REGISTER_ADDRESS				0xE5
#define W_INTERNAL_REGISTER						0xE6
#define R_INTERNAL_REGISTER						0xE7
#define W_INTERNAL_MAILBOX_ADDRESS				0xE8
#define W_INTERNAL_MAILBOX						0xE9
#define R_INTERNAL_MAILBOX						0xEA
#define W_EXT_PAD_ADDRESS						0xEB
#define W_EXT_PAD_DATA							0xEC
#define R_EXT_PAD_DATA							0xED

#endif /*DPP343x_H_*/
