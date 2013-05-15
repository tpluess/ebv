/* Copying and distribution of this file, with or without modification,
 * are permitted in any medium without royalty. This file is offered as-is,
 * without any warranty.
 */

/*! @file template_ipc.h
 * @brief Shared header file between application and its
 * CGI. Contains all information relevant to IPC between these two.
 */

#ifndef TEMPLATE_IPC_H_
#define TEMPLATE_IPC_H_

/* The parameter IDs to identify the different requests/responses. */
enum EnIpcParamIds
{
	GET_APP_STATE,
	GET_NEW_IMG,
	SET_IMAGE_TYPE,
	SET_EXPOSURE_TIME,
	SET_THRESHOLD
};

/*! @brief The path of the unix domain socket used for IPC between the application and its user interface. */
#define USER_INTERFACE_SOCKET_PATH "/tmp/IPCSocket.sock"

/*! @brief Describes a rectangular sub-area of an image. */
struct IMG_RECT
{
	/*! @brief Rectangle width. */
	uint16 width;
	/*! @brief Rectangle height. */
	uint16 height;
	/*! @brief X Coordinate of the lower left corner.*/
	uint16 xPos;
	/*! @brief Y Coordinate of the lower left corne.*/
	uint16 yPos;
};

/*! @brief The different modes the application can be in. */
enum EnAppMode
{
	APP_OFF,
	APP_CAPTURE_ON
};

/*! @brief Object describing all the state information the web interface needs to know about the application. */
struct APPLICATION_STATE
{
	/*! @brief Whether a new image is ready to display by the web interface. */
	bool bNewImageReady;
	/*! @brief The time stamp when the last live image was taken. */
	uint32 imageTimeStamp;
	/*! @brief The mode the application is running in. Depending on the mode different information may have to be displayed on the web interface.*/
	enum EnAppMode enAppMode;
	/*! @brief the image type index */
	unsigned int nImageType;
	/*! @brief Shutter time in micro seconds.*/
	int nExposureTime;
	/*! @brief cut off value for change detection.*/
	int nThreshold;
	/*! @brief  the step counter */
	unsigned int nStepCounter;
  
  
  int objectcount;
  unsigned int thres_calc;
};

#endif /*TEMPLATE_IPC_H_*/
