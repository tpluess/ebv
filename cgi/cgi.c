/* Copying and distribution of this file, with or without modification,
 * are permitted in any medium without royalty. This file is offered as-is,
 * without any warranty.
 */

/*! @file cgi_template.c
 * @brief CGI used for the webinterface of the SHT application.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#include "cgi.h"

#include <time.h>

/*! @brief Main object structure of the CGI. Contains all 'global'
 * variables. */
struct CGI_TEMPLATE cgi;

/*! @brief All potential arguments supplied to this CGI. */
struct ARGUMENT args[] =
{
	{ "exposureTime", INT_ARG, &cgi.args.nExposureTime, &cgi.args.bExposureTime_supplied },
	{ "Threshold", INT_ARG, &cgi.args.nThreshold, &cgi.args.bThreshold_supplied },
	{ "ImageType", INT_ARG, &cgi.args.nImageType, &cgi.args.bImageType_supplied },
  { "objectcount", INT_ARG, &cgi.args.nbrofobj, &cgi.args.nbrofobj_supp },
  { "threscalc", INT_ARG, &cgi.args.threshcalc, &cgi.args.threshcalc_supp }
};

/*! @brief Strips whiltespace from the beginning and the end of a string and returns the new beginning of the string. Be advised, that the original string gets mangled! */
char * strtrim(char * str) {
	char * end = strchr(str, 0) - 1;

	while (*str != 0 && strchr(" \t\n", *str) != NULL)
		str += 1;

	while (end > str && strchr(" \t\n", *end) != NULL)
		end -= 1;

	*(end + 1) = 0;

	return str;
}

/*********************************************************************//*!
 * @brief Split the supplied URI string into arguments and parse them.
 *
 * Matches the argument string with the arguments list (args) and fills in
 * their values. Unknown arguments provoke an error, but missing
 * arguments are just ignored.
 *
 * @param strSrc The argument string.
 * @param srcLen The length of the argument string.
 * @return SUCCESS or an appropriate error code otherwise
 *//*********************************************************************/
static OSC_ERR CGIParseArguments()
{
	char buffer[1024];

	/* Intialize all arguments as 'not supplied' */
	for (int i = 0; i < sizeof args / sizeof (struct ARGUMENT); i += 1)
	{
		*args[i].pbSupplied = false;
	}

	while (fgets (buffer, sizeof buffer, stdin)) {
		struct ARGUMENT *pArg = NULL;
		char * key, * value = strchr(buffer, ':');

		if (value == NULL) {
			OscLog(ERROR, "%s: Invalid line: \"%s\"\n", __func__, buffer);
			return -EINVALID_PARAMETER;
		}

		*value = 0;
		value += 1;

		key = strtrim(buffer);
		value = strtrim(value);

		OscLog(INFO, "obtained key: %s, and Value: %s\n", key, value);

		for (int i = 0; i < sizeof(args)/sizeof(struct ARGUMENT); i += 1) {
			if (strcmp(args[i].strName, key) == 0) {
				pArg = args + i;
				break;
			}
		}

		if (pArg == NULL) {
			OscLog(ERROR, "%s: Unknown argument encountered: \"%s\"\n", __func__, key);
			return -EINVALID_PARAMETER;
		} else {
			if (pArg->enType == STRING_ARG) {
				// FIXME: Could someone fix this buffer overflow?
				strcpy((char *) pArg->pData, value);
			} else if (pArg->enType == INT_ARG) {
				if (sscanf(value, "%d", (int *) pArg->pData) != 1) {
					OscLog(ERROR, "%s: Unable to parse int value of variable \"%s\" (%s)!\n", __func__, pArg->strName, value);
					return -EINVALID_PARAMETER;
				}
			} else if (pArg->enType == SHORT_ARG) {
				if (sscanf(value, "%hd", (short *) pArg->pData) != 1) {
					OscLog(ERROR, "%s: Unable to parse short value of variable \"%s\" (%s)!\n", __func__, pArg->strName, value);
					return -EINVALID_PARAMETER;
				}
			} else if (pArg->enType == BOOL_ARG) {
				if (strcmp(value, "true") == 0) {
					*((bool *) pArg->pData) = true;
				} else if (strcmp(value, "false") == 0) {
					*((bool *) pArg->pData) = false;
				} else {
					OscLog(ERROR, "CGI %s: Unable to parse boolean value of variable \"%s\" (%s)!\n", __func__, pArg->strName, value);
					return -EINVALID_PARAMETER;
				}
			}

			if (pArg->pbSupplied != NULL)
				*pArg->pbSupplied = true;
		}
	}

	return SUCCESS;
}

/*********************************************************************//*!
 * @brief Query the current state of the application and see what else
 * we need to get from it
 *
 * Depending on the current state of the application, other additional
 * parameters may be queried.
 *
 * @return SUCCESS or an appropriate error code otherwise
 *//*********************************************************************/
static OSC_ERR QueryApp()
{
	OSC_ERR err;
	struct OSC_PICTURE pic;

	/* First, get the current state of the algorithm. */
	err = OscIpcGetParam(cgi.ipcChan, &cgi.appState, GET_APP_STATE, sizeof(struct APPLICATION_STATE));
	if (err != SUCCESS)
	{
		/* This request is defined in all states, and thus must succeed. */
		OscLog(ERROR, "CGI: Error querying application! (%d)\n", err);
		return err;
	}

	switch(cgi.appState.enAppMode)
	{
	case APP_OFF:
		/* Algorithm is off, nothing else to do. */
		break;
	case APP_CAPTURE_ON:
		if (cgi.appState.bNewImageReady)
		{
			/* If there is a new image ready, request it from the application. */
			err = OscIpcGetParam(cgi.ipcChan, cgi.imgBuf, GET_NEW_IMG, OSC_CAM_MAX_IMAGE_WIDTH/2*OSC_CAM_MAX_IMAGE_HEIGHT/2);
			if (err != SUCCESS)
			{
				OscLog(DEBUG, "CGI: Getting new image failed! (%d)\n", err);
				return err;
			}

			/* Write the image to the RAM file system where it can be picked
			 * up by the webserver on request from the browser. */
			pic.width = OSC_CAM_MAX_IMAGE_WIDTH/2;
			pic.height = OSC_CAM_MAX_IMAGE_HEIGHT/2;
			pic.type = OSC_PICTURE_GREYSCALE;
			pic.data = (void*)cgi.imgBuf;

			return OscBmpWrite(&pic, IMG_FN);
		}
		break;
	default:
		OscLog(ERROR, "%s: Invalid application mode (%d)!\n", __func__, cgi.appState.enAppMode);
		break;
	}
	return SUCCESS;
}

/*********************************************************************//*!
 * @brief Set the parameters for the application supplied by the web
 * interface.
 *
 * @return SUCCESS or an appropriate error code otherwise
 *//*********************************************************************/
static OSC_ERR SetOptions()
{
	OSC_ERR err;
	struct ARGUMENT_DATA *pArgs = &cgi.args;

	if (pArgs->bImageType_supplied)
	{
		err = OscIpcSetParam(cgi.ipcChan, &pArgs->nImageType, SET_IMAGE_TYPE, sizeof(pArgs->nImageType));
		if (err != SUCCESS)
		{
			OscLog(DEBUG, "CGI: Error setting option! (%d)\n", err);
			return err;
		}
	}

	if (pArgs->bThreshold_supplied)
	{
		err = OscIpcSetParam(cgi.ipcChan, &pArgs->nThreshold, SET_THRESHOLD, sizeof(pArgs->nThreshold));
		if (err != SUCCESS)
		{
			OscLog(DEBUG, "CGI: Error setting option! (%d)\n", err);
			return err;
		}
	}

	if (pArgs->bExposureTime_supplied)
	{
		err = OscIpcSetParam(cgi.ipcChan, &pArgs->nExposureTime, SET_EXPOSURE_TIME, sizeof(pArgs->nExposureTime));
		if (err != SUCCESS)
		{
			OscLog(DEBUG, "CGI: Error setting option! (%d)\n", err);
			return err;
		}
	}

	return SUCCESS;
}

/*********************************************************************//*!
 * @brief Take all the gathered info and formulate a valid AJAX response
 * that can be parsed by the Javascript in the browser.
 *//*********************************************************************/
static void FormCGIResponse()
{
	struct APPLICATION_STATE  *pAppState = &cgi.appState;

	/* Header */
	printf("Content-type: text/plain\n\n" );

	printf("imgTS: %u\n", (unsigned int)pAppState->imageTimeStamp);
	printf("exposureTime: %d\n", pAppState->nExposureTime);
	printf("Threshold: %d\n", pAppState->nThreshold);
	printf("Stepcounter: %d\n", pAppState->nStepCounter);
	printf("width: %d\n", OSC_CAM_MAX_IMAGE_WIDTH/2);
	printf("height: %d\n", OSC_CAM_MAX_IMAGE_HEIGHT/2);
	printf("ImageType: %u\n", pAppState->nImageType);
  printf("objectcount: %u\n", pAppState->objectcount);
  printf("threscalc: %u\n", pAppState->thres_calc);

	fflush(stdout);
}

OscFunction(mainFunction)
	OSC_ERR err;
	struct stat socketStat;

	/* Initialize */
	memset(&cgi, 0, sizeof(struct CGI_TEMPLATE));

	/* First, check if the algorithm is even running and ready for IPC
	 * by looking if its socket exists.*/
	if(stat(USER_INTERFACE_SOCKET_PATH, &socketStat) != 0)
	{
		/* Socket does not exist => Algorithm is off. */
		/* Form a short reply with that info and shut down. */
		cgi.appState.enAppMode = APP_OFF;
		OscFail_m("Algorithm is off!");
		return -1;
	}

	/******* Create the framework **********/
	OscCall(OscCreate,
		&OscModule_log,
		&OscModule_ipc);

	OscLogSetConsoleLogLevel(CRITICAL);
	OscLogSetFileLogLevel(DEBUG);

	OscCall( OscIpcRegisterChannel, &cgi.ipcChan, USER_INTERFACE_SOCKET_PATH, 0);

	OscCall( CGIParseArguments);

	/* The algorithm negative acknowledges if it cannot supply
	 * the requested data, i.e. it changed state during the
	 * process of getting the data.
	 * Try again until we succeed. */
	do
	{
		do
		{
			err = QueryApp();
		} while (err == -ENEGATIVE_ACKNOWLEDGE);

		OscAssert_m( err == SUCCESS, "Error querying algorithm!");
		err = SetOptions();
	} while (err == -ENEGATIVE_ACKNOWLEDGE);
	FormCGIResponse();

	OscDestroy();

OscFunctionCatch()
	OscDestroy();
	OscLog(INFO, "Quit application abnormally!\n");
OscFunctionEnd()

	/*********************************************************************//*!
	 * @brief Execution starting point
	 *
	 * Handles initialization, control and unloading.
	 * @return 0 on success, -1 otherwise
	 *//*********************************************************************/
int main(void) {
	if (mainFunction() == SUCCESS)
		return 0;
	else
		return 1;
}

