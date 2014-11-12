#include "FaceDetector.h"



FaceDetector::FaceDetector( IplImage* image, bool isColor ) : m_Image(image), m_bIsColor(isColor), m_NewImage(NULL)
{
    m_Cascade = NULL;
    m_Cascade = (CvHaarClassifierCascade*)cvLoad( HAAR_CASCADE_FRONTAL_FILENAME.c_str(),0,0,0);

    if ( !m_Cascade )
        throw std::string("FaceDetector constructor could not create cascade.  Check path?");

    if ( !m_Image )
        throw std::string("FaceDetector needs an image to work on");
}

FaceDetector::~FaceDetector()
{
    Reset();
}

int FaceDetector::Detect(bool bOnlyFindLargest)
{
    int nFaces = 0;

    CvMemStorage* storage = NULL;
    storage = cvCreateMemStorage(0);
    cvClearMemStorage(storage);
    CvPoint pt1, pt2;
    // IplImage* tempimg = NULL;
    CvSeq* faces = NULL;
    CvSize minFeatureSize = cvSize(20, 20);   // don't find faces less than 20x20


    // If it is set, the function uses Canny edge detector to reject some image regions that contain too few or too many edges
    // and thus can not contain the searched object
    int flags = CV_HAAR_DO_CANNY_PRUNING;
    if ( bOnlyFindLargest )
        flags |= CV_HAAR_FIND_BIGGEST_OBJECT;

    faces = cvHaarDetectObjects(m_Image,m_Cascade,storage,1.1,2,flags);
    if ( faces )
    {
        if ( !bOnlyFindLargest )
        {
            nFaces = faces->total;

            for ( int i = 0; i < nFaces; i++ )
            {
                CvRect* r = (CvRect*)cvGetSeqElem(faces,i);
                m_Rects.push_back(r);
            }
        }
        else
        {
            if ( faces->total > 0 )
            {
                nFaces = 1;
                CvRect* r = (CvRect*)cvGetSeqElem(faces,0);
                m_Rects.push_back(r);
            }
        }


        // now draw the rectanges on the new image
        m_NewImage = cvCreateImage(cvSize(m_Image->width,m_Image->height),8,(m_bIsColor ? 3 : 1));

        if ( !m_NewImage )
            throw std::string("FaceDetector::Detect could not create new image");

        cvCopy(m_Image,m_NewImage,NULL);

        for ( size_t i = 0; i < m_Rects.size(); i++ )
        {
            pt1.x = m_Rects[i]->x;
            pt2.x = m_Rects[i]->x+m_Rects[i]->width;
            pt1.y=m_Rects[i]->y;
            pt2.y=m_Rects[i]->y+m_Rects[i]->height;

            cvRectangle(m_NewImage, pt1, pt2, CV_RGB(255,0,0), 3,8,0);  // draw red rectangle

            // create face for storage
            IplImage* tempface = NULL;
            tempface = cvCreateImage(cvSize(m_Rects[i]->width,m_Rects[i]->height),m_Image->depth, m_Image->nChannels);

            if ( !tempface )
                throw std::string("FaceDetector::Detect could not create new face image");

            cvSetImageROI( m_Image, *m_Rects[i] );
            cvCopy(m_Image,tempface);
            m_Faces.push_back(tempface);
            cvResetImageROI( m_Image );
        }
    }
    return nFaces;
}



void FaceDetector::Reset()
{
    if ( m_NewImage )
        cvReleaseImage(&m_NewImage);

    m_NewImage = NULL;

    for ( ImageVec::iterator it = m_Faces.begin(); it != m_Faces.end(); it++ )
        cvReleaseImage(&(*it));

    m_Rects.clear();
    m_Faces.clear();
}

