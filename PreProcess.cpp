#include "PreProcess.h"
#include "FaceDetector.h"

/*
Function:   DetectAndPreProcess
Purpose:    Given a file and a name, try to find the face in the image (largest face)
Save this file as the name to disk
Notes:
Throws      std::string if somthing goes wrong
Returns:    true if face found and data saved, false if no face found
*/
bool DetectAndPreProcess(const char *image, const char* name)
{
    IplImage* faceImage = NULL;
    bool bRes = false;

    // try to open the given file containing the face
    // the one indicates that we assume the image is color
    faceImage = cvLoadImage(image,1);

    if ( image )
    {
        try
        {
            FaceDetector* fd = new FaceDetector(faceImage, true);

            // find the largest face in the image
            fd->Detect(true);
            IplImage *tempFace;

            // did we find the face?
            if ( !fd->GetFaceVec().empty() )
            {
                // get the face from the face detector
                IplImage* face = fd->GetFaceVec()[0];

                // now perform the rest of the preprocessing on the face
                tempFace = cvCreateImage(cvSize(face->width, face->height), face->depth, 1);
                ConvertToGreyScale(face, tempFace);
                Resize(face, tempFace);

                // do histogram equalization on the found face
                cvEqualizeHist(tempFace, tempFace);

                // try to save it to disk
                if ( !cvSaveImage( name, tempFace ) )
                {
                    std::string err;
                    err = "Error: DetectAndPreProcess could not save ";
                    err += image;
                    err += " as ";
                    err += name;
                    throw err;
                }

                bRes = true;

            }

            delete fd;

        }
        catch (...)
        {
            throw;
        }

    }
    else
    {
        // could not open image
        std::string err;
        err = "Error: DetectAndPreProcess could not open ";
        err += image;
        throw err;
    }

    return bRes;

}




/*
Function:   PreProcess
Purpose:    Resize, make into grey scale, and do histogram equalization on given image
Notes:
Throws      std::string if somthing goes wrong
*/

void PreProcess( IplImage* src, IplImage** dest )
{
    if ( *dest )
        cvReleaseImage(&*dest);

    try
    {
        FaceDetector* fd = new FaceDetector(src, false);
        fd->Detect(true);

        if ( !fd->GetFaceVec().empty() )
        {
            // get the face from the face detector
            IplImage* face = fd->GetFaceVec()[0];

            int width = 100;
            int height = 100;

            *dest = cvCreateImage(cvSize(width, height), src->depth, 1);
            if ( !*dest )
                throw std::string("PreProcess could not create dest image");

            if ( src->nChannels != 1 )
                ConvertToGreyScale(face, face);

            Resize(face, *dest);

            // do histogram equalization on the found face
            cvEqualizeHist(*dest, *dest);
        }
        else
        {
            throw std::string("FaceDetector could not find face");
        }
    }
    catch ( ... )
    {
        throw;
    }
}

