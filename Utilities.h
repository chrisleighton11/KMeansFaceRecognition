#ifndef UTILITIES_H
#define UTILITIES_H

/*
   Utilties.h
   Description:   defines useful functions
   Author:        Chris Leighton
   Date:          Feb 10th 2010

*/

#include <string>
#include <iostream>
#include <sstream>

#include <cv.h>
#include <cvaux.h>
#include <cxcore.h>
#include <highgui.h>


typedef std::vector<CvRect*>        RectVec;
typedef std::vector<IplImage*>      ImageVec;


// convert image to grey scale
void ConvertToGreyScale(const IplImage* input, IplImage* output);

// convert float image to grey scale
IplImage* ConvertFloatToGreyScale( const IplImage* image );


// perform histogram equalization on image
void HistogramEqualization(const IplImage* input, IplImage* output);


// resize image
void Resize(const IplImage* input, IplImage* output);


// get string representing the date and time
std::string getDateTime();


/*

how to get the time is takes to do somthing in ms
int ms
t = (double)cvGetTickCount();
t = (double)cvGetTickCount() - t;
ms = cvRound( t / ((double)cvGetTickFrequency() * 1000.0) );
*/

#endif
