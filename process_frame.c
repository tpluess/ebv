#include "template.h"
#include <string.h>
#include <stdlib.h>

#define IMAGE(a) (data.u8TempImage[a])
#define SZ (sizeof(IMAGE(GRAYSCALE)))
#define NC (OSC_CAM_MAX_IMAGE_WIDTH / 2u)

#define LABEL_COLOR 0x80

OSC_ERR draw_bbox(struct OSC_PICTURE *picIn, struct OSC_VIS_REGIONS *regions, uint8 Color);
static void imhist(uint32_t hist[], const uint8_t* image);
static void erosion(uint8_t* dst, const uint8_t* src);
static void dilation(uint8_t* dst, const uint8_t* src);
static void thresh(uint8_t* dst, const uint8_t* src, uint8_t threshold);
static void region_labeling(uint8_t* dst);
static uint8_t otsu(const uint8_t* img);

void ProcessFrame(uint8 *pInputImg)
{
  uint8_t thres = (uint8_t)data.ipc.state.nThreshold;
  
  /* if the threshold is 0, the otsu algorithm is used */
  if(thres == 0)
  {    
    thres = otsu(IMAGE(GRAYSCALE));
  }
  
  data.ipc.state.thres_calc = thres;

  /* make the threshold image */
  thresh(IMAGE(THRESHOLD), IMAGE(GRAYSCALE), thres);
  
  /* dilation */
  dilation(IMAGE(DILATION), IMAGE(THRESHOLD));
  
  /* erosion */
  erosion(IMAGE(EROSION), IMAGE(DILATION));
  
  /* do the region labeling stuff */
  region_labeling(IMAGE(LABELIMG));
}



/* Drawing FuNCtion for Bounding Boxes; own implementation because Oscar only allows colored boxes; here in Gray value "Color"  */
/* should only be used for debugging purposes because we should not drawn into a gray scale image */
OSC_ERR draw_bbox(struct OSC_PICTURE *picIn, struct OSC_VIS_REGIONS *regions, uint8 Color)
{
	 uint16 i, o;
	 uint8 *pImg = (uint8*)picIn->data;
	 const uint16 width = picIn->width;
	 for(o = 0; o < regions->noOfObjects; o++)//loop over regions
	 {
		 /* Draw the horizontal lines. */
		 for (i = regions->objects[o].bboxLeft; i < regions->objects[o].bboxRight; i += 1)
		 {
				 pImg[width * regions->objects[o].bboxTop + i] = Color;
				 pImg[width * (regions->objects[o].bboxBottom - 1) + i] = Color;
		 }

		 /* Draw the vertical lines. */
		 for (i = regions->objects[o].bboxTop; i < regions->objects[o].bboxBottom-1; i += 1)
		 {
				 pImg[width * i + regions->objects[o].bboxLeft] = Color;
				 pImg[width * i + regions->objects[o].bboxRight] = Color;
		 }
	 }
	 return SUCCESS;
}

static void imhist(uint32_t hist[], const uint8_t* image)
{
  uint32_t idx;
  
  /* loop through the whole image; don't care a rats arse whether these
     are rows or columns because only the grey value is interesting */
  for(idx = 0; idx < SZ; idx++)
  {
    /* this counts the grey values */
    hist[*image] += 1;
    
    /* next pixel */
    image++;
  }
}

static void erosion(uint8_t* dst, const uint8_t* src)
{
  uint32_t r, c;

  /* loop through all rows, but skop 1st and last one */
  for(r = NC; r < SZ - NC; r+= NC)
  {
    /* loop through all columns, but skop 1st and last one */
    for(c = 1; c < NC - 1; c++)
    {
      /* shorthand */
      const uint8_t* ptr = &src[r + c];
      
      /* this is the actual erosion */
      IMAGE(EROSION)[r + c] =
        *(ptr - NC - 1u)  & *(ptr + NC)   & *(ptr - NC + 1u)  &
        *(ptr - 1)        & *ptr          & *(ptr + 1)        &
        *(ptr + NC - 1)   & *(ptr + NC)   & *(ptr + NC + 1);
    }
  }
}

static void dilation(uint8_t* dst, const uint8_t* src)
{
  uint32_t c, r;
  
  /* loop through all rows, but skop 1st and last one */
  for(r = NC; r < SZ - NC; r+= NC)/* we skip the first and last line */
  {
    /* loop through all columns, but skop 1st and last one */
    for(c = 1; c < NC - 1; c++)
    {
      /* shorthand */
      const uint8_t* ptr = &src[r+c];
      
      /* this is the actual dilation */
      IMAGE(DILATION)[r+c] =
        *(ptr - NC - 1u)  | *(ptr - NC)   | *(ptr - NC + 1)   |
        *(ptr - 1)        | *ptr          | *(ptr + 1)        |
        *(ptr + NC - 1)   | *(ptr + NC)   | *(ptr + NC + 1);
    }
  }
}

static void thresh(uint8_t* dst, const uint8_t* src, uint8_t threshold)
{
  uint32_t c, r;
  
  /* loop through all rows, but skop 1st and last one */
  for(r = 0; r < SZ; r+= NC)
  {
    /* loop through all columns, but skop 1st and last one */
    for(c = 0; c < NC; c++)
    {
      /* shorthand */
      const uint8_t tmp = src[r + c];
      
      /* check the threshold */
      if(threshold < tmp)
      {
        dst[r + c] = 0u;
      }
      else
      {
        dst[r + c] = 255u;
      }
    }
  }
}

static void region_labeling(uint8_t* dst)
{
  uint8_t tmp_buf[SZ];
  
  /* input picture info */
  struct OSC_PICTURE picin =
  {
    .data = IMAGE(DILATION),
    .width = NC,
    .height = OSC_CAM_MAX_IMAGE_HEIGHT / 2u,
    .type = OSC_PICTURE_GREYSCALE
  };

  /* temporary binary image */
  struct OSC_PICTURE picout =
  {
    .data = tmp_buf,
    .width = NC,
    .height = OSC_CAM_MAX_IMAGE_HEIGHT / 2u,
    .type = OSC_PICTURE_BINARY
  };
  
  /* region labeling info */
  struct OSC_VIS_REGIONS region_info;
  
  /* make a binary image */
  OscVisGrey2BW(&picin, &picout, 0x80, false);
  
  /* now do the region labeling */
  OscVisLabelBinary(&picout, &region_info);
  
  /* feature extraction */
  OscVisGetRegionProperties(&region_info);
  
  /* now copy the grayscale image to the labeling image */
  memcpy(dst, IMAGE(GRAYSCALE), SZ);
  
  /* abuse the picin picture info to draw the bounding boxes */
  picin.data = dst;
  draw_bbox(&picin, &region_info, 0x80);
  
  /* set the number of objects in the web gui */
  data.ipc.state.objectcount = region_info.noOfObjects;
}

static uint8_t otsu(const uint8_t* img)
{
  uint8_t ret;
  uint32_t histogram[256];
  uint32_t k, g;
  uint32_t w0, w1;
  uint32_t mu0s, mu1s;
  float mu0, mu1;
  float sigma_b, sigma_max;

  /* initialise the histogram with zero */
  memset(histogram, 0, sizeof(histogram));

  /* calculate the histogram */
  imhist(histogram, img);
  
  sigma_max = 0.0f;
  
  for(k = 0; k < 256; k++)
  {
    w0 = w1 = 0;
    mu0s = mu1s = 0;
    for(g = 0; g < 256; g++)
    {
      if(g <= k)
      {
        w0 += histogram[g];
        mu0s += histogram[g] * g;
      }
      else
      {
        w1 += histogram[g];
        mu1s += histogram[g] * g;
      }
    }
    mu0 = ((float)mu0s / (float)w0);
    mu1 = ((float)mu1s / (float)w1);
    
    sigma_b = ((float)(w0 * w1)) * (mu0 - mu1) * (mu0 - mu1);
    if(sigma_b > sigma_max)
    {
      sigma_max = sigma_b;
      ret = (uint8_t)k;
    }
  }
  
  return ret;
}

