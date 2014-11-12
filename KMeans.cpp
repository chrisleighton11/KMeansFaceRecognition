#include "KMeans.h"
#include <fstream>


void KMeans1(const std::string& databaseName, std::ofstream& out)
{
    if ( !out.is_open() )
        throw std::string("Kmeans1 - results file not open");

    try
    {
        //open database
        Database db;
        db.Read(databaseName);

        // members of Database class that we will need (The rest is external)
        Database::ImageVec& imageVec = db.GetImageVec();
        int nImages = db.GetnImages();
        int nPeople = db.GetnPeople();

        // get dimensions of original images
        CvSize imageCvSize;
        size_t imageSize;
        imageCvSize.width = imageVec[0].m_Image->width;
        imageCvSize.height = imageVec[0].m_Image->height;
        imageSize = imageCvSize.width * imageCvSize.height;

        // store original images in CvMat - each row contains a width*height image
        // also initialize each label as person ID so cvKmeans2 can use these as the initial approximation
        CvMat* originalImages = cvCreateMat(nImages, imageSize, CV_32FC1);
        CvMat* labels = cvCreateMat(nImages, 1, CV_32SC1);

        for ( int i = 0; i < nImages; i++ )
        {
            IplImage* img = imageVec[i].m_Image;
            labels->data.i[i] = personIDMatrix->data.i[i] - 1;

            // iterate each image, store intensity in originalImages
            for ( int row = 0; row < img->height; row++ )
            {
                for ( int col = 0; col < img->width; col++ )
                {
                    CvScalar s;
                    s = cvGet2D(img, row, col);
                    int colInOriginals = row*img->height+col;
                    int rowInOriginals = i*imageSize;
                    int posInOriginals = rowInOriginals + colInOriginals;
                    originalImages->data.fl[posInOriginals] = (double)s.val[0];
                }
            }
        }

        // now do kmeans on the images
        CvTermCriteria crit = cvTermCriteria( CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 10, 1000.0 );
        int attempts = 100;
        CvMat* centers = cvCreateMat(nImages, 1, CV_32FC1);

        cvKMeans2(projectedFaceMatrix, nPeople, labels, crit, attempts, NULL, CV_KMEANS_USE_INITIAL_LABELS, centers);

        out << "Results for KMeans on original images" << std::endl;
        for ( int i = 0; i < nImages; i++ )
        {
            out << "Image ID: " << personIDMatrix->data.i[i] << " Clusters to: " << (labels->data.i[i] + 1) << "Cluster Center: " << centers->data.fl[i] << std::endl;
        }
    }
    catch ( ... )
    {
        throw;
    }

}
