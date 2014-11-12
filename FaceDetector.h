/*
   FaceDetector.cpp
   Description:   Class used to detect faces in an image, can be used to find all faces or
                  just the largest
   Author:        Chris Leighton
   Date:          Feb 10th 2010

*/

#ifndef FACEDETECTOR_H
#define FACEDETECTOR_H

#include <iostream>
#include <string>
#include <vector>


#include <cv.h>
#include <cvaux.h>
#include <cxcore.h>
#include <highgui.h>

#include "Utilities.h"

static std::string HAAR_CASCADE_FRONTAL_FILENAME = "C:\\Program Files\\OpenCV2.2\\data\\haarcascades\\haarcascade_frontalface_alt.xml";
static std::string HAAR_CASCADE_PROFILE_FILENAME = "//root//OpenCV-2.2.0//data//haarcascades//haarcascade_profileface.xml";

class FaceDetector
{
public:
    FaceDetector( IplImage* image, bool isColor );
    ~FaceDetector();

    int Detect(bool bOnlyFindLargest = false); // detect faces in image, return number of faces found

    const IplImage* GetNewImage()
    {
        return m_NewImage;
    }
    const RectVec   GetRectVec()
    {
        return m_Rects;
    }
    const ImageVec  GetFaceVec()
    {
        return m_Faces;
    }


private:
    IplImage*                 m_Image;
    bool                       m_bIsColor;
    CvHaarClassifierCascade*   m_Cascade;

    // results
    // store the CvRects representing the faces
    RectVec                    m_Rects;     // stored coordinates for where we found the faces
    IplImage*                  m_NewImage;  // same as old image with rectangles drawn on it
    ImageVec                   m_Faces;     // each face found
    // Get ready for different image
    void Reset();

};



#endif

