#ifndef PREPROCESS_H
#define PREPROCESS_H

/*
   Preprocess.h
   Description:   Provides function for preprocessing a face in an image
   Author:        Chris Leighton
   Date:          Feb 16th 2010

*/

#include "Utilities.h"

bool DetectAndPreProcess(const char *image, const char* name);
void PreProcess( IplImage* src, IplImage** dest );


#endif

