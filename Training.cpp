#include "Training.h"
#include "PreProcess.h"
#include <fstream>
#include "HTMLHelper.h"


/*
   Function:   Train
   Purpose:    Calls into the Trainer to train the system and load the database
   Notes:      generates html results to show what happened
   Throws
*/
void Train(const char* imagelist, const char* database, std::string& resultdir)
{
    try
    {
        Trainer trn(imagelist,database);
        trn.LoadImages();
        trn.CreateSubspace();
        trn.ProjectOntoSubSpace();
        trn.CalculateThresholds();
        trn.MakeDatabase();
        // trn.GenResults(resultdir);

    }
    catch (...)
    {
        throw;
    }


}




////////////////////////////////////////////
//           Trainer class                //
////////////////////////////////////////////


/*
   Function:   Trainer constructor
   Purpose:
   Notes:
   Throws
*/
Trainer::Trainer(const char* imagelist, const char* database)
{
    m_ImageFile = imagelist;
    m_DatabaseFile = database;

    m_pDatabase = new Database();

}



/*
   Function:   Trainer destructor
   Purpose:    destructor for the trainer, cleans up all of the memory
   Notes:
   Throws
*/
Trainer::~Trainer()
{
    if ( m_pDatabase )
        delete m_pDatabase;
}



/*
   Function:   LoadImages
   Purpose:    reads in m_ImageFile and loads the images
   Notes:      LoadImages will pre-process the images
   Throws      std::string if file can not be opened, or if image can not be found
   returns:    Number if images processed
*/
int Trainer::LoadImages()
{
    // open the iamges file
    std::ifstream in(m_ImageFile.c_str());

    if ( !in.is_open() )
    {
        std::string err;
        err = "Trainer could not open images file ";
        err += m_ImageFile;
        throw err;
    }

    // buffer to read each line of file
    char buffer[512];

    Database::NameVec& names = m_pDatabase->GetNames();
    Database::ImageVec& imageVec = m_pDatabase->GetImageVec();
    int nImages = 0;
    while ( in.getline(buffer,512) )
    {
        try
        {
            std::string line(buffer);
            if ( line.empty() )
                break;

            Image img(buffer);

            if ( img.m_ID == 0 )
                throw std::string("Trainer::LoadImages - Training person ids should start with 1");

            names.push_back(img.m_PersonName);

            // load image
            IplImage* temp = NULL;
            temp = cvLoadImage(img.m_ImageName.c_str(),CV_LOAD_IMAGE_GRAYSCALE);

            if ( !temp )
            {
                std::string err;
                err = "Trainer::LoadImages could not create image for ";
                err += img.m_ImageName;
                throw err;
            }

            IplImage* newtemp = NULL;

            PreProcess(temp, &newtemp);

            //////////////////////////img.m_Image = newtemp;
            img.m_Image = newtemp;

            // save this information
            imageVec.push_back(img);

            ////////////////////////////cvReleaseImage(&temp);
        }
        catch (...)
        {
            throw;
        }

        // success
        nImages++;
    }

    m_pDatabase->SetnImages(nImages);
    in.close();

    // store images and person id's in array to pass to eigen functions
    imageArray = (IplImage**)cvAlloc(nImages*sizeof(IplImage*));

    personIDMatrix  = cvCreateMat(1,nImages,CV_32SC1);

    for ( int i = 0; i < nImages; i++ )
    {
        personIDMatrix->data.i[i] = imageVec[i].m_ID;
        std::cout << "Adding person id: " << personIDMatrix->data.i[i] << std::endl;

        imageArray[i] = imageVec[i].m_Image;
    }

    return nImages;
}



/*
   Function:   CreateSubspace
   Purpose:    finds average image, centers each image around mean, finds covariance matrix, then finds
	       Eigenvectors (Principal components) and Eigenvalues
   Notes:
   Throws      std::string if it can't allocate memory
   returns:
*/

void Trainer::CreateSubspace()
{
    // we can only find m_nImages - 1 eigenvalues
    int nImages = m_pDatabase->GetnImages();
    int nEigenVals = nImages - 1;
    m_pDatabase->SetnEigenVals(nEigenVals);
    Database::ImageVec imageVec = m_pDatabase->GetImageVec();

    CvSize size;
    size.width = imageVec[0].m_Image->width;
    size.height = imageVec[0].m_Image->height;

    // allocate space for the eigen vectors
    eigenVectorArray = (IplImage**)cvAlloc(sizeof(IplImage*) * nEigenVals);
    for ( int i = 0; i < nEigenVals; i++ )
    {
        eigenVectorArray[i] = cvCreateImage(size, IPL_DEPTH_32F, 1);   // floating point image

        if ( !eigenVectorArray[i] )
            throw std::string("Trainer::DoPCA could not allocate EigenVector");
    }


    averageImage = cvCreateImage(size, IPL_DEPTH_32F, 1 );
    if ( !averageImage )
        throw std::string("Trainer::DoPCA could not allocate AverageImage");

    // This is how many eigen vectors we will use out of all of them
    // i.e in the subspace, each image will Ŝhave nEigenVals
    CvTermCriteria limit = cvTermCriteria(CV_TERMCRIT_ITER, nEigenVals, 1);

    // these will be the actuall eigen values
    eigenValueMatrix = cvCreateMat(1, nEigenVals, CV_32FC1);

    // ask openCv to do the work
    cvCalcEigenObjects( nImages, (void*)imageArray, (void*)eigenVectorArray, CV_EIGOBJ_NO_CALLBACK, 0, 0, &limit,
                        averageImage, eigenValueMatrix->data.fl );

    // now we have the averge image, eigenvectors of the covariance matrix, and eigen values
    cvNormalize(eigenValueMatrix, eigenValueMatrix, 1, 0, CV_L1, 0);

}




/*
   Function:   ProjectOntoSubspace
   Purpose:    projects the faces onto the PCA subspace
   Notes:
   Throws
   returns:
*/
void Trainer::ProjectOntoSubSpace()
{
    int nImages = m_pDatabase->GetnImages();
    int nEigenVals = m_pDatabase->GetnEigenVals();

    projectedFaceMatrix = cvCreateMat(nImages, nEigenVals, CV_32FC1);

    // calculate each images projection onto the eigen subspace
    // m_ProjectedFaceMatrix->data.fl + (row*nEigenVals) gets us to the row containing the images
    // projection
    // m_ProjectedFaceMatrix is a [m_nImages][nEigenVals] matrix

    // to avoid getting a bunch of NaN values, I normalize the Eigenvalues to be between 0 and 1
    cvNormalize(eigenValueMatrix, eigenValueMatrix, 1, 0, CV_L1, 0);

    for ( int row = 0; row < nImages; row++ )
    {
        cvEigenDecomposite(imageArray[row], nEigenVals, eigenVectorArray, 0, 0, averageImage,
                           projectedFaceMatrix->data.fl + (row*nEigenVals));
    }

    // now the training projection is completed, Each row of m_ProjectedFaceMatrix represents
    // each image's values projected onto the new subspace.
    // in other words, where each image used to be a NxM matrix, it is now only nEigenVals long
    // Thus, we have reduced dimentionality... Yeah
}



/*
   Function:   GenResults
   Purpose:    create images and html representation of results of training session
   Notes:
   Throws      std::string if it can't create directory
   returns:    void
*/
void Trainer::GenResults(std::string& resultsdir)
{
    /*CvMat* projectedFaceMatrix = m_pDatabase->GetProjectedFaceMatrix();
    IplImage** eigenVectorArray = m_pDatabase->GetEigenVectorArray();
    CvMat* eigenValueMatrix = m_pDatabase->GetEigenValueMatrx();
    int nImages = m_pDatabase->GetnImages();
    int nEigenVals = m_pDatabase->GetnEigenVals();
    IplImage** imageArray = m_pDatabase->GetImageArray();
    IplImage* averageImage = m_pDatabase->GetAverageImage();
    Database::NameVec

    // first get the name of the results.  Name it the database name

    std::string resultsname = m_DatabaseFile.substr(0, m_DatabaseFile.size()-4); // chop of .xml

    // get rid of prefix directory since we are storing results in fiven resultsdir
    size_t dirpos = resultsname.find_last_of('/');
    if ( dirpos != std::string::npos )
        resultsname = resultsname.substr(dirpos+1, resultsname.size() - dirpos );

    std::string htmlfile = resultsdir + resultsname + ".html"; // name the reults after the database created
    std::string imageprefix = resultsdir + resultsname;


    std::ofstream html(htmlfile.c_str());
    if ( !html.is_open() )
    {
        std::string err;
        err = "Trainer::GenResults could not open html file ";
        err += htmlfile;
        throw err;
    }

    html << GetHeader();
    html << GetTitle(resultsname.c_str());

    html << GetText("Training Results", "h1", "    ");

    std::stringstream s1;
    s1 << "Number of Images " << nImages;
    html << GetText(s1.str().c_str(), "h2", "    ");

    std::vector<std::string> people;
    for ( size_t i = 0; i < Names.size(); i++ )
    {
        std::vector<std::string>::iterator it;
        if ( (it = find(people.begin(), people.end(), m_Names[i])) == people.end() )
        {
            people.push_back(m_Names[i]);
        }
    }

    std::stringstream s2;
    s2 << "Number of people " << people.size();
    html << GetText(s2.str().c_str(), "h2", "    ");

    html << GetText("Names:", "h2", "    ");
    for ( size_t i = 0; i < people.size(); i++ )
    {
        html << GetText(people[i].c_str(), "h4", "    ");
        for ( int j = 0; j < m_nImages; j++ )
        {
            if ( m_ImageVec[j].m_PersonName == people[i] )  //hack
            {
                std::string savename = m_ImageVec[j].m_ImageName;
                size_t p = savename.find_last_of('/');
                if ( p != std::string::npos )
                    savename = savename.substr(p+1, savename.size()-p);
                p = savename.find_last_of('.');
                if ( p != std::string::npos )
                    savename = savename.substr(0, savename.size()-4);
                std::stringstream hname;
                hname << people[i] << "_" << savename << ".jpg";
                std::string sname = resultsdir;
                sname += hname.str();
                cvSaveImage(sname.c_str(), m_ImageVec[j].m_Image);
                html << GetImageTag(hname.str().c_str(), "200", "200", "    ");
            }
        }
    }

    std::stringstream s3;
    s3 << "Number of Eigen Values " << m_nEigenVals;
    html << GetText(s3.str().c_str(), "h2", "    ");

    // create images of average and some of the eigenfaces
    std::string avgname = imageprefix + "_average.jpg";
    cvSaveImage(avgname.c_str(), m_AverageImage);

    html << GetText("Average Image", "h2", "   ");
    html << GetBr("   ");

    // only output name, not entire directory
    // to make this simple I am going to put everything in the same directory
    std::string tempname = "";
    size_t pos = 0;
    pos = avgname.find_last_of('/');
    if ( pos != std::string::npos )
        tempname = avgname.substr(pos+1, avgname.size()-pos);
    else
        tempname = avgname;
    html << GetImageTag(tempname.c_str(), "200", "200", "   ");

    html << GetBr("    ");

    // now create some of the eigen vectors
    // for now I'll just do five
    html << GetText("Some EigenFaces", "h2", "   ");
    int n = m_nEigenVals;
    for ( int i = 0; i < n && i < 5; i++ )
    {
        // make name of image to save to disk
        std::stringstream s5;
        s5 << imageprefix << "_EigenFace_" << i << ".jpg";
        std::string name = s5.str();
        pos = name.find_last_of('/');
        if ( pos != std::string::npos )
            tempname = name.substr(pos+1, name.size()-pos);

        IplImage* greyimage = ConvertFloatToGreyScale(m_EigenVectorArray[i]);
        cvSaveImage(name.c_str(), greyimage);
        cvReleaseImage(&greyimage);

        html << GetImageTag(tempname.c_str(), "200", "200", "   ");

        html << GetBr("    ");
    }

    html << GetClosingTags();

    html.close();*/
}



/*
   Function:   CalculateThresholds
   Purpose:    calculates thresholds of database used for comparing a probe image
   Notes:
   Throws
   returns:    void
*/
void Trainer::CalculateThresholds()
{
    int nImages = m_pDatabase->GetnImages();
    int nEigenVals = m_pDatabase->GetnEigenVals();

    double maxE = 0.0;
    double maxM = 0.0;

    int index = 0;

    while ( index <= nImages )
    {

        for ( int row = index; row < nImages; row++ )
        {
            double e_distance = 0.0;
            double m_distance = 0.0;

            for ( int col = 0; col < nEigenVals; col++ )
            {
                double d = projectedFaceMatrix->data.fl[index*nEigenVals +col] - projectedFaceMatrix->data.fl[row*nEigenVals + col];

                double dd = d*d;
                e_distance += dd;
                m_distance += dd/eigenValueMatrix->data.fl[col];
            }

            if ( e_distance > maxE )
                maxE = e_distance;
            if ( m_distance > maxM )
                maxM = m_distance;
        }

        index++;
    }
    m_pDatabase->SetEuclideanThreshold(maxE * .5);
    m_pDatabase->SetMahalanobisThreshold(maxM * .5);

}


void Trainer::MakeDatabase()
{
    m_pDatabase->Write(m_DatabaseFile);
}
