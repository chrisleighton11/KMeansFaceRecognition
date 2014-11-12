/*

EigenFace.cpp
Description:   Driver for detecting faces using EigenFace with Principal Component Analysis
	           provides comand line user interface to interact with prgram

Author:        Chris Leighton
Date:          Feb 10th 2010

*/

#ifndef EIGENFACE_TEST

#include <algorithm>
#include <cctype>
#include <string>
#include <cv.h>
#include <cvaux.h>
#include <cxcore.h>
#include <highgui.h>
#include "FaceDetector.h"
#include "Utilities.h"
#include "PreProcess.h"
#include "Training.h"
#include "TrainingFile.h"
#include "Recognize.h"


void PrintUsage();

int main( int argc, char** argv )
{
    system("clear");
    std::string command = "";

    while (1)
    {
        PrintUsage();

        cin >> command;

        std::transform( command.begin(), command.end(), command.begin(),(int(*)(int)) std::toupper );

        try
        {

            if ( command == "PREPROCESS" )
            {
                std::string input;
                std::string output;
                cout << "Enter Input image name: ";
                cin >> input;
                cout << "Enter new name of preprocessed image: ";
                cin >> output;
                if ( DetectAndPreProcess(input.c_str(), output.c_str()) )
                    cout << "Detected face in " << input << ". PreProcessed face saved as " << output << endl;
                else
                    cout << "An error occured attemting to detect a face and PreProcess " << input  << endl;
            }
            else if ( command == "GENFILE" )
            {
                std::string trainingfile;
                std::string basedir;
                cout << "Enter Training File Name:";
                cin >> trainingfile;

                cout << "Enter base directory:";
                cin >> basedir;

                TrainingFile tf( trainingfile );
                tf.SetBaseDir(basedir);

                std::string name = "";
                std::string filename = "";

                while (1)
                {
                    // get lines from user
                    cout << "Enter Person Name (q to quit):";
                    cin >> name;
                    if ( name == "q" || name == "Q" )
                        break;

                    cout << "Enter File Name:";
                    cin >> filename;

                    tf.addEntry(name, filename);

                    name = filename = "";
                }

                if ( tf.GenFile() )
                {
                    cout << trainingfile << " created." << endl;
                }
                else
                {
                    cout << "An error occured" << endl;
                }
            }
            else if ( command == "TRAIN" )
            {
                std::string trainingfile;
                std::string outputfile;
                std::string resultsdir;
                cout << "Enter Training File:";
                cin >> trainingfile;
                cout << "Enter database name:";
                cin >> outputfile;
                cout << "Enter results directory:";
                cin >> resultsdir;

                // make sure results dir ends with '/'
                if ( !resultsdir.empty() &&  resultsdir[resultsdir.size()-1] != '/' )
                    resultsdir.append("/");

                Train( trainingfile.c_str(), outputfile.c_str(), resultsdir );

                cout << "Database created: " << outputfile << endl;

            }
            else if ( command == "SEARCH" )
            {
                std::string imagename;
                std::string database;
                std::string resultsdir;
                cout << "Enter image to search for: ";
                cin >> imagename;
                cout << "Enter trained database file name: ";
                cin >> database;
                cout << "Enter results directory:";
                cin >> resultsdir;

                // make sure results dir ends with '/'
                if ( !resultsdir.empty() &&  resultsdir[resultsdir.size()-1] != '/' )
                    resultsdir.append("/");

                double distance = DBL_MAX;
                int idFound = 0;
                std::string result = Recognize(imagename.c_str(),database.c_str(), distance, resultsdir, idFound );
                if ( result.empty() )
                {
                    cout << "Could not find person" << endl;
                    cout << "Distance: " << distance << endl;
                }
                else
                {
                    cout << "Found: " << result << endl;
                    cout << "Distance: " << distance << endl;
                }
            }
            else if ( command == "SYS" )
            {
                // if ( bAllowSys )  // this just makes it easy to demo the program
                // this should be turned off for security reasons
                // under normal operations
                {
                    char command[256];
                    cin.getline(command, 256);
                    system(command);
                }
            }

            else if ( command == "EXIT" )
            {
                cout << "good by" << endl;
                break;
            }

            command = "";
        }
        catch ( std::string err )
        {
            cout << "Error: " << err << endl;
            return 1;
        }
    }

    return 0;
}




void PrintUsage()
{
    cout << "Please select a command:" << endl << endl;
    cout << "preprocess - detect a face and preprocess the image, then store face on disk" << endl;
    cout << "genfile    - create a training file" << endl;
    cout << "train      - train the system" << endl;
    cout << "search     - search the database for a face in an image" << endl;
    cout << "exit" << endl << ":";
}

#endif // EIGENFACE_TEST
