#include "Utilities.h"
#include <time.h>


/*
   Function:   ConvertToGrayScale
   Purpose:    Converts color image to grey scale
   Notes:
   Throws
*/
void ConvertToGreyScale(const IplImage* input, IplImage* output)
{
    if ( !input )
        throw std::string("ConvertToGreyScale received null image as argument");

    if ( input->nChannels > 1 )
    {
        cvCvtColor( input, output, CV_BGR2GRAY );
    }
    else
    {
        output = cvCloneImage(input);
    }
}


/*
   Function:   HistogramEqualization
   Purpose:    Performs Histogram Equalization
   Notes:
   Throws
*/
void HistogramEqualization(const IplImage* input, IplImage* output)
{
    cvEqualizeHist(input, output);
}



/*
   Function:   Resize
   Purpose:    Resizes image
   Notes:      Does not keep apect ratio the same
   Throws
*/
void Resize(const IplImage* input, IplImage* output)
{
    int flag = 0;

    if (output->width < input->width && output->height < input->height )
        flag |= CV_INTER_AREA; // shrinking
    else
        flag |= CV_INTER_LINEAR;

    cvResize(input, output, flag);
}


/*
   Function: ConvertFloatToGreyScale
   Purpose:  Given a float image, convert it to grey scale
   Notes:    users should release returned image themselves
   Throws:   std::string if it can't create the new image
   Returns:  Grey Scale Image
*/
IplImage* ConvertFloatToGreyScale( const IplImage* image )
{
    IplImage* result = NULL;

    double min, max;

    // get the min and max values with help from openCV
    cvMinMaxLoc(image, &min, &max);

    // deal with NaN
    if (cvIsNaN(min) || min < -1e30)
        min = -1e30;
    if (cvIsNaN(max) || max > 1e30)
        max = 1e30;
    if (max-min == 0.0f)
        max = min + 0.001;

    result = cvCreateImage(cvSize(image->width, image->height), 8, 1);
    if ( !result )
        throw std::string("ConvertFloatToGreyScale could not create image.");

    // thankyou openCV
    cvConvertScale(image, result, 255.0 / (max - min), - min * 255.0 / (max-min));

    return result;
}



std::string getDateTime()
{
    time_t tim=time(NULL);
    tm *now=localtime(&tim);

    char thedate[50];
    char thetime[50];

    sprintf(thedate, "%02d/%02d/%d", now->tm_mon, now->tm_mday, now->tm_year+1900 );
    sprintf(thetime, "%02d:%02d:%02d", now->tm_hour, now->tm_min, now->tm_sec);

    std::string ret(thedate);
    ret += " ";
    ret += thetime;

    return ret;
}

