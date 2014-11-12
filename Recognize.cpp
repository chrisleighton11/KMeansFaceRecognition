#include "Recognize.h"
#include "FaceDetector.h"
#include "PreProcess.h"
#include <fstream>
#include "HTMLHelper.h"

/*
   Function:   Recognize
   Purpose:    Recognize a face
   Arguments:  1) the image with the face to recognize 2) the trained database
   Notes:      Function will return empty string if we don't find the person
   Returns:    std::string with persons name we found
   Throws:     std::string if it can't open file or create memory
*/
std::string Recognize( const char* image, const char* database, double& distance, std::string& resultsDir, int& idFound, Database* db, bool bCheckDistance )
{
    std::string personFound = "";
    try
    {
        if ( db )
        {
            Recognizer r(db, image, database);
            // find the person
            std::string foundPerson = "";

            idFound = 0;
            personFound = r.FindFace(0, distance, idFound, bCheckDistance);
        }
        else
        {
            Recognizer r(image, database);
            r.LoadTrainingDatabase();

            // find the person
            std::string foundPerson = "";

            idFound = 0;
            personFound = r.FindFace(0, distance, idFound, bCheckDistance);
        }
    }
    catch (...)
    {
        throw;
    }

    return personFound;
}



/*
   Function:   Recognizer class constructor
   Purpose:
   Arguments:  1) the image with the face to recognize 2) the trained database
   Notes:
   Throws:     std::string if it can't open file or create memory
*/
Recognizer::Recognizer( const char* imagename, const char* databasename ) : m_DatabaseName(databasename), m_pDatabase(NULL), m_SearchImageName(imagename),
                        m_FaceImage(NULL), m_nFacesToFind(0), m_IDFound(0), m_DistanceFound(0.0), m_PersonFound(""), m_bDeleteDb(true)
{
    IplImage* tempface = cvLoadImage(imagename,CV_LOAD_IMAGE_GRAYSCALE);
    PreProcess(tempface, &m_FaceImage);
    if ( m_FaceImage )
    {
        m_nFacesToFind = 1;
        m_FacesToFind = (IplImage**)cvAlloc(m_nFacesToFind*sizeof(IplImage*));
        m_FacesToFind[0] = m_FaceImage;

        m_pDatabase = new Database();
    }
    else
    {
        std::string err;
        err = "Recognizer could not load image: ";
        err += imagename;
        throw err;
    }
}


Recognizer::Recognizer( Database* db, const char* imagename, const char* databasename) : m_DatabaseName(databasename), m_SearchImageName(imagename),
                        m_FaceImage(NULL), m_nFacesToFind(0), m_IDFound(0), m_DistanceFound(0.0), m_PersonFound(""), m_bDeleteDb(false)
{
    IplImage* tempface = cvLoadImage(imagename,CV_LOAD_IMAGE_GRAYSCALE);
    PreProcess(tempface, &m_FaceImage);
    if ( m_FaceImage )
    {
        m_nFacesToFind = 1;
        m_FacesToFind = (IplImage**)cvAlloc(m_nFacesToFind*sizeof(IplImage*));
        m_FacesToFind[0] = m_FaceImage;
        m_pDatabase = db;
    }
    else
    {
        std::string err;
        err = "Recognizer could not load image: ";
        err += imagename;
        throw err;
    }
}



/*
   Function:   Recognizer class destructor
   Purpose:    clean up memory that Recognizer uses
   Notes:
*/
Recognizer::~Recognizer()
{
    // release the image with all of the faces
    cvReleaseImage(&m_FaceImage);

    if ( m_bDeleteDb && m_pDatabase )
        delete m_pDatabase;
}




/*
   Function:   LoadTrainingDatabase
   Purpose:    loads to training database that was creating during the training session
   Notes:
   Returns:    true if success
*/
bool Recognizer::LoadTrainingDatabase()
{
    return m_pDatabase->Read(m_DatabaseName);
}



/*
   Function:   FindFace
   Purpose:    attempts to find a face in the database
   Notes:      this function uses distance to determine how
               confident we are with the closest face, the threshold value can be adjusted
               to try to prevent false positive results
   Returns:    name of person, if it finds it, empty if it does not find the face
   throws:
*/
std::string Recognizer::FindFace( int faceNum, double& distance, int& idFound, bool bCheckDistance )
{
    std::string personName = "";
    int nEigenVals = m_pDatabase->GetnEigenVals();
    int nImages = m_pDatabase->GetnImages();
    int nPeople = m_pDatabase->GetnPeople();
    Database::NameVec& namesVec = m_pDatabase->GetNames();
    double mahalanobisThreshold = m_pDatabase->GetMahalanobisThreshold();
    double euclideanThreshold = m_pDatabase->GetEuclideanThreshold();

    // project the test face onto
    // the PCA subspace so try to find a match
    if ( faceNum < 0 || faceNum >= m_nFacesToFind )
        throw std::string("Recognizer::FindFace - Invalid face number argument");

    float *projectedFace = NULL;  // this is the face that results from projecting the new face onto the subspace
    projectedFace = (float*)cvAlloc(nEigenVals*sizeof(float));

    cvEigenDecomposite(m_FacesToFind[faceNum], nEigenVals, eigenVectorArray, 0, 0, averageImage, projectedFace );

    int 		e_index = 0; // index that results from using EuclideanDistance
    int 		m_index = 0; // index that results from using MahalanobisDistance
    double 	    e_distance = 0.0;
    double      m_distance = 0.0;

    e_index = EuclideanDistance(projectedFace, e_distance);
    m_index = MahalanobisDistance(projectedFace, m_distance);

    // select the lowest distance
    int index = 0;  // the index that we will use
    distance = DBL_MAX;

    if ( bCheckDistance )
    {
        bool bGoodDistance = false;
        /*if ( e_distance <= euclideanThreshold )
        {
             index = e_index;
             distance = e_distance;
             bGoodDistance = true;
        std::cout << "Using Euclidean Distance" << std::endl;
        } ////////////////// Not using Euclidean Distance - too many false positive results
        else*/
        if ( m_distance <= mahalanobisThreshold )
        {
            index = m_index;
            distance = m_distance;
            bGoodDistance = true;
        }
        if ( bGoodDistance )
        {
            // we have an acceptable match
            // return the persons name
            int id = personIDMatrix->data.i[index];

            personName = namesVec[index];
            m_IDFound = id;
            idFound = id;
        }
    }
    else
    {
        int id = personIDMatrix->data.i[m_index];
        personName = namesVec[m_index];
        distance = m_distance;
        m_IDFound = id;
        idFound = id;
    }
    m_PersonFound = personName;
    m_DistanceFound = distance;



    ///////////////////CLUSTER TEST CODE///////////////////

    // kmeans2 -
    // samples = prejectedFaceMatrix
    // nclusters = nPeople

    /*CvMat* clusters = cvCreateMat(nImages, 1, CV_32SC1);
    CvTermCriteria crit = cvTermCriteria( CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 10000, 1.0 );


    // calculate means for each projected face
    double *projectedMeans = new double[nImages];
    for ( int row = 0; row < nImages; row++ )
    {
        double total = 0.0;
        for ( int col = 0; col < nEigenVals; col++ )
        {
            total += projectedFaceMatrix->data.fl[row*nEigenVals+col];
        }

        projectedMeans[row] = total/(double)nEigenVals;
        //cout << "Mean for image " << row+1 << " ID: " << personIDMatrix->data.i[row] << " is: " << projectedMeans[row] << std::endl;
    }


    // output projected values to disk (transposed)
    std::ofstream out("projectedValues.csv");
    CvMat* projectedFaceMatrixT = cvCreateMat(nEigenVals, nImages, CV_32FC1);
    cvTranspose(projectedFaceMatrix, projectedFaceMatrixT);
    for ( int row = 0; row < nEigenVals; row++ )
    {
        for ( int col = 0; col < nImages; col++ )
        {
            out << projectedFaceMatrixT->data.fl[row*nImages+col] << ", ";
        }
        out << std::endl;
    }
    cvReleaseMat(&projectedFaceMatrixT);
    out.close();


    cvKMeans2(projectedFaceMatrix, nPeople, clusters, crit);

    /*for ( int i = 0; i < nImages; i++ )
    {
        std::cout << "Image ID: " << personIDMatrix->data.i[i] << " Clusters to: " << (clusters->data.i[i] + 1) << std::endl;
    }

    delete projectedMeans; */
    return personName;
}




/*
   Function:   EuclideanDistance
   Purpose:    find the closest image and subsequent person name for a given face
   Notes:
   Returns:    person ID of person found and populates distance argument
   throws:
*/
int Recognizer::EuclideanDistance( float* projectedTestFace, double& distance )
{
    double bestChoiceDiff = DBL_MAX;
    int bestIndex = 0;
    int nEigenVals = m_pDatabase->GetnEigenVals();
    int nImages = m_pDatabase->GetnImages();

    for ( int row = 0; row < nImages; row++ )
    {
        double distance = 0.0;

        for ( int col = 0; col < nEigenVals; col++ )
        {
            // subtract each projected face's coefficient value to find out
            // how close they are
            float d = projectedTestFace[col] - projectedFaceMatrix->data.fl[row*nEigenVals + col];
            distance += d*d;
        }

        if ( distance < bestChoiceDiff )
        {
            bestChoiceDiff = distance;
            bestIndex = row;
        }
    }

    distance = bestChoiceDiff;

    return bestIndex;
}




/*
   Function:  MahalanobisDistance
   Purpose:   find closest image and person using Mahalanobis distance
   Notes:     fills in distance
   Returns:   index of person found

*/
int Recognizer::MahalanobisDistance( float* projectedTestFace, double& distance )
{
    double bestChoiceDiff = DBL_MAX;
    int bestIndex = 0;
    int nEigenVals = m_pDatabase->GetnEigenVals();
    int nImages = m_pDatabase->GetnImages();

    for ( int row = 0; row < nImages; row++ )
    {
        double distance = 0.0;

        for ( int col = 0; col < nEigenVals; col++ )
        {
            // subtract each eigenvector value to find out
            // how close they are
            float d = projectedTestFace[col] - projectedFaceMatrix->data.fl[row*nEigenVals + col];

            distance += d*d / eigenValueMatrix->data.fl[col];
        }
        distance = sqrt(distance);

        if ( distance < bestChoiceDiff )
        {
            bestChoiceDiff = distance;
            bestIndex = row;
        }
    }

    distance = bestChoiceDiff;

    return bestIndex;
}



/*
function:	GenResults
Purpose:	Generate html and image results for face search
Notes:
*/
void Recognizer::GenResults(std::string& resultsDir)
{
    /*
    // lets name the results after the image we searched for
    // first I need to get just the image name and remove the directories
    std::string searchImageName(m_SearchImageName);

    // get rid of directory
    size_t pos = 0;
    pos = searchImageName.find_last_of('/');
    if ( pos != std::string::npos )
        searchImageName = searchImageName.substr(pos+1, searchImageName.size()-pos);

    // now change the extension from .xxx to _xxx
    std::string extension = searchImageName.substr(searchImageName.size()-3, 3);
    std::string saveImageName = resultsDir + searchImageName;   // to write image to new directory
    searchImageName = searchImageName.substr(0, searchImageName.size()-4);
    searchImageName += "_";
    searchImageName += extension;

    std::string resultsname = resultsDir + searchImageName;

    std::string htmlname = resultsname + ".html";

    std::ofstream html(htmlname.c_str());

    if ( !html.is_open() )
    {
        std::string err;
        err = "GenResults can not open results file ";
        err += htmlname;
        throw err;
    }

    html << GetHeader();
    html << GetTitle(searchImageName.c_str());
    std::string title;
    title = "Search Results for ";
    title += searchImageName;
    html << GetText(title.c_str(),"h1", "    ");


    // IE and firefox will not show some image formats.. e.g .pgm
    // I'll make it .jpg.  openCV automatically saves in the format based on
    // file extension.
    saveImageName = saveImageName.substr(0, saveImageName.size()-4); // chop extension
    saveImageName += ".jpg"; // add new extension

    cvSaveImage(saveImageName.c_str(), m_FaceImage);
    html << GetText("Search Face", "h2", "    ");
    std::string tempname = ""; // cut out directory, assume all files in same location
    pos = saveImageName.find_last_of('/');
    if ( pos != std::string::npos )
        tempname = saveImageName.substr(pos+1, saveImageName.size()-pos);
    else
        tempname = saveImageName;

    html << GetImageTag(tempname.c_str(), "200", "200", "    ");

    if ( m_IDFound )
    {
        std::stringstream s1;
        s1 << "Search Found Person ID: " << m_IDFound;
        html << GetText(s1.str().c_str(), "h3", "    ");

        std::stringstream s2;
        s2 << "PersonFound: " << m_PersonFound;
        html << GetText(s2.str().c_str(), "h3", "    ");

        std::stringstream s3;
        s3 << "Distance: " << m_DistanceFound;
        html << GetText(s3.str().c_str(), "h3", "    ");

        html << GetText("Images in database with this persons face:", "h3", "    ");

        // save original images to load into html
        for ( int i = 0; i < m_nImages; i++ )
        {
            if ( m_PersonIDMatrix->data.i[i] == m_IDFound )
            {
                std::string original = m_OriginalImages[i];
                // load the original
                IplImage* image = cvLoadImage(original.c_str(), CV_LOAD_IMAGE_GRAYSCALE);
                if ( image )
                {
                    // create new name
                    pos = 0;
                    pos = original.find_last_of('/');  // cut off directory
                    if ( pos != std::string::npos )
                        original = original.substr(pos+1, original.size()-pos);

                    original = original.substr(0, original.size()-4);
                    original += ".jpg"; // change type

                    // insert personfound name - if there is one
                    if ( !m_PersonFound.empty() )
                    {
                        original.insert(0, "_");
                        original.insert(0, m_PersonFound);
                    }

                    std::string saveimage = resultsDir + original;

                    // save to disk
                    cvSaveImage(saveimage.c_str(), image);
                    cvReleaseImage(&image);

                    std::stringstream htmlname;
                    htmlname << original << " database index: " << i;
                    html << GetText(htmlname.str().c_str(), "h3", "    ");
                    html << GetImageTag(original.c_str(), "200", "200", "    " );
                }
            }
        }

    }
    else
    {
        html << GetText("Search failed to find match", "h3", "    ");

        std::stringstream s1;
        s1 << "Distance: " << m_DistanceFound;
        html << GetText(s1.str().c_str(), "h3", "    ");
    }


    html << GetClosingTags();
    html.close();

*/


}
