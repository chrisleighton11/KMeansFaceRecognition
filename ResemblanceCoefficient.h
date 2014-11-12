#ifndef RESEMBLANCECOEFFICIENT_H
#define RESEMBLANCECOEFFICIENT_H


#include <limits>
#include <cmath>
#include "Utilities.h"
#include  "Cluster.h"



// Below are methods for the resemblance coefficients
// The cluster Techniques should be declared as classes that support the methods:
//     GetDataMatrix() and GetResemblanceCoefficientMatrix() and int GetnObjects()
// each method below will populate the resemblance matrix appropriatly
// it is also important to disinguish between similarity and dissimilarity coefficients when
// clustering which is why IsDissimilarType exists



enum ResemblanceCoefficientType
{
    BrayCurtisCoefficient,
    CanberraMetricCoefficient,
    CoefficientOfShapeDiff,
    CorrelationCoefficient,
    CosineCoefficient,
    EuclideanDistanceCoefficient,
    MahalanobisDistanceCoefficient
};


inline bool IsDisimilarType( ResemblanceCoefficientType t )
{
    if ( BrayCurtisCoefficient == t ||
         CanberraMetricCoefficient == t ||
         CoefficientOfShapeDiff == t ||
         EuclideanDistanceCoefficient == t ||
         MahalanobisDistanceCoefficient == t )
        return true;

    return false;
}


///////////////////////////////////////////////////////////////////////////
//////////////////// BrayCurtisCoefficient ////////////////////////////////
///////////////////////////////////////////////////////////////////////////

template <typename T>
void CalcBrayCurtisCoefficient(T* obj, int nObjects)
{
    CvMat* dataMatrix = obj->GetDataMatrix();
    CvMat* resemblanceMatrix = obj->GetResemblanceMatrix();
    int nAttributes = obj->GetnAttributes();

    if ( !dataMatrix || !resemblanceMatrix || nObjects < 2 || nAttributes < 1 )
        throw std::string("BrayCurtisCoefficient needs data");

    // initialize resemblanceMatrix
    for ( int i = 0; i < nObjects*nObjects; i++ )
        resemblanceMatrix->data.fl[i] = (obj->IsDisimilarityCoeffcient() ? std::numeric_limits<double>::max()
                                         : -1 * std::numeric_limits<double>::max());

    // do the calculations for the each unique combination of objects
    int row_it = 1;
    while ( row_it < nObjects )
    {
        for ( int col_it = 0; col_it < row_it; col_it++ )  // only do calcualation for one direction
        {
            int j = row_it;
            int k = col_it;
            double bjk = 0.0;

            double numerator = 0.0;
            double denominator = 0.0;

            for ( int col = 0; col < nAttributes; col++ )
            {
                double valj = dataMatrix->data.fl[j*nAttributes+col];
                double valk = dataMatrix->data.fl[k*nAttributes+col];

                numerator += abs(valj - valk);
                denominator += valj + valk;
            }
            if ( 0.0 != denominator )
                bjk = numerator / denominator;

            resemblanceMatrix->data.fl[row_it*nObjects+col_it] = bjk;
        }

        row_it++;
    }
}


///////////////////////////////////////////////////////////////////////////
//////////////////// CanberraMetricCoefficient ////////////////////////////
///////////////////////////////////////////////////////////////////////////

template <typename T>
void CalcCanberraMetricCoefficient(T* obj, int nObjects)
{
    CvMat* dataMatrix = obj->GetDataMatrix();
    CvMat* resemblanceMatrix = obj->GetResemblanceMatrix();
    int nAttributes = obj->GetnAttributes();

    if ( !dataMatrix || !resemblanceMatrix || nObjects < 2 || nAttributes < 1 )
        throw std::string("CalcCanberraMetricCoefficient needs data");

    // initialize resemblanceMatrix
    for ( int i = 0; i < nObjects*nObjects; i++ )
        resemblanceMatrix->data.fl[i] = (obj->IsDisimilarityCoeffcient() ? std::numeric_limits<double>::max()
                                         : -1 * std::numeric_limits<double>::max());

    // do the calculations for the each unique combination of objects
    int row_it = 1;
    while ( row_it < nObjects )
    {
        for ( int col_it = 0; col_it < row_it; col_it++ )  // only do calcualation for one direction
        {
            int j = row_it;
            int k = col_it;
            double ajk = 0.0;
            double n = (double)nAttributes;
            double total = 0.0;

            for ( int col = 0; col < nAttributes; col++ )
            {
                double valj = dataMatrix->data.fl[j*nAttributes+col];
                double valk = dataMatrix->data.fl[k*nAttributes+col];

                double numerator = abs(valj - valk);
                double denominator = valj + valk;
                if ( 0.0 != denominator )
                    total += numerator / denominator;
            }
            ajk = (1/n) * total;

            resemblanceMatrix->data.fl[row_it*nObjects+col_it] = ajk;
        }

        row_it++;
    }
}



///////////////////////////////////////////////////////////////////////////
//////////////////// CoefficientOfShapeDiff ///////////////////////////////
///////////////////////////////////////////////////////////////////////////

template <typename T>
void CalcCoefficientOfShapeDiff(T* obj, int nObjects)
{
    CvMat* dataMatrix = obj->GetDataMatrix();
    CvMat* resemblanceMatrix = obj->GetResemblanceMatrix();
    int nAttributes = obj->GetnAttributes();

    if ( !dataMatrix || !resemblanceMatrix || nObjects < 2 || nAttributes < 1 )
        throw std::string("CalcCoefficientOfShapeDiff needs data");

    // initialize resemblanceMatrix
    for ( int i = 0; i < nObjects*nObjects; i++ )
        resemblanceMatrix->data.fl[i] = (obj->IsDisimilarityCoeffcient() ? std::numeric_limits<double>::max()
                                         : -1 * std::numeric_limits<double>::max());

    // do the calculations for the each unique combination of objects
    int row_it = 1;
    while ( row_it < nObjects )
    {
        for ( int col_it = 0; col_it < row_it; col_it++ )  // only do calcualation for one direction
        {
            int j = row_it;
            int k = col_it;
            double zjk = 0.0;
            double djk = 0.0;
            double qjk = 0.0;
            double n = (double)nAttributes;

            // need to find euclidean distance between object j and k
            // and find sum of all attributes for each object j and k
            double j_sum = 0.0;
            double k_sum = 0.0;

            for ( int col = 0; col < nAttributes; col++ )
            {
                double dis =  dataMatrix->data.fl[j*nAttributes+col] - \
                            dataMatrix->data.fl[k*nAttributes+col];
                djk += dis * dis;

                j_sum += dataMatrix->data.fl[j*nAttributes+col];
                k_sum += dataMatrix->data.fl[k*nAttributes+col];
            }
            djk /= n;
            qjk = ( 1 / (n*n) ) * ( (j_sum - k_sum)*(j_sum - k_sum) );
            zjk = ( n / (n-1) ) * ( djk - qjk );
            zjk = sqrt(zjk);

            resemblanceMatrix->data.fl[row_it*nObjects+col_it] = sqrt(zjk);
        }

        row_it++;
    }
}



///////////////////////////////////////////////////////////////////////////
//////////////////// CorrelationCoefficient ///////////////////////////////
///////////////////////////////////////////////////////////////////////////

template <typename T>
void CalcCorrelationCoefficient(T* obj, int nObjects)
{
    CvMat* dataMatrix = obj->GetDataMatrix();
    CvMat* resemblanceMatrix = obj->GetResemblanceMatrix();
    int nAttributes = obj->GetnAttributes();

    if ( !dataMatrix || !resemblanceMatrix || nObjects < 2 || nAttributes < 1 )
        throw std::string("CalcCorrelationCoefficient needs data");

    // initialize resemblanceMatrix
    for ( int i = 0; i < nObjects*nObjects; i++ )
        resemblanceMatrix->data.fl[i] = (obj->IsDisimilarityCoeffcient() ? std::numeric_limits<double>::max()
                                         : -1 * std::numeric_limits<double>::max());

    // do the calculations for the each unique combination of objects
    int row_it = 1;
    while ( row_it < nObjects )
    {
        for ( int col_it = 0; col_it < row_it; col_it++ )  // only do calcualation for one direction
        {
            int j = row_it;
            int k = col_it;
            double rjk = 0.0;
            double XijXik = 0.0;
            double Xij = 0.0;
            double Xij2 = 0.0;
            double Xik = 0.0;
            double Xik2 = 0.0;
            double oneOverN = 1 / (double)nAttributes;

            for ( int col = 0; col < nAttributes; col++ )
            {
                double valj = dataMatrix->data.fl[j*nAttributes+col];
                double valk = dataMatrix->data.fl[k*nAttributes+col];

                XijXik += ( valj * valk );
                Xij += valj;
                Xij2 += ( valj * valj );
                Xik += valk;
                Xik2 += ( valk * valk );
            }

            double numerator = XijXik - ( oneOverN * Xij * Xik );

            double denom_p1 = Xij2 - ( oneOverN * (Xij * Xij) );
            double denom_p2 = Xik2 - ( oneOverN * (Xik * Xik) );
            double denominator = sqrt( denom_p1 * denom_p2 );

            if ( denominator == 0 )
                rjk = 0.0;
            else
                rjk = numerator / denominator;
            resemblanceMatrix->data.fl[row_it*nObjects+col_it] = rjk;
        }

        row_it++;
    }
}


///////////////////////////////////////////////////////////////////////////
//////////////////// CosineCoefficient ////////////////////////////////////
///////////////////////////////////////////////////////////////////////////

template <typename T>
void CalcCosineCoefficient(T* obj, int nObjects)
{
    CvMat* dataMatrix = obj->GetDataMatrix();
    CvMat* resemblanceMatrix = obj->GetResemblanceMatrix();
    int nAttributes = obj->GetnAttributes();

    if ( !dataMatrix || !resemblanceMatrix || nObjects < 2 || nAttributes < 1 )
        throw std::string("CalcCosineCoefficient needs data");

    // initialize resemblanceMatrix
    for ( int i = 0; i < nObjects*nObjects; i++ )
        resemblanceMatrix->data.fl[i] = (obj->IsDisimilarityCoeffcient() ? std::numeric_limits<double>::max()
                                         : -1 * std::numeric_limits<double>::max());

    // do the calculations for the each unique combination of objects
    int row_it = 1;
    while ( row_it < nObjects )
    {
        for ( int col_it = 0; col_it < row_it; col_it++ )  // only do calcualation for one direction
        {
            int j = row_it;
            int k = col_it;
            double cjk = 0.0;
            double XijXik = 0.0;
            double Xij = 0.0;
            double Xik = 0.0;

            for ( int col = 0; col < nAttributes; col++ )
            {
                double valj = dataMatrix->data.fl[j*nAttributes+col];
                double valk = dataMatrix->data.fl[k*nAttributes+col];
                XijXik += ( valj * valk );
                Xij += ( valj * valj );
                Xik += ( valk * valk );
            }
            cjk = XijXik / (sqrt(Xij) * sqrt(Xik));

            resemblanceMatrix->data.fl[row_it*nObjects+col_it] = cjk;
        }

        row_it++;
    }
}


///////////////////////////////////////////////////////////////////////////
//////////////////// Euclidean  ///////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////

template <typename T>
void CalcEuclideanDistanceCoefficient(T* obj, int nObjects)
{
    CvMat* dataMatrix = obj->GetDataMatrix();
    CvMat* resemblanceMatrix = obj->GetResemblanceMatrix();
    int nAttributes = obj->GetnAttributes();

    if ( !dataMatrix || !resemblanceMatrix || nObjects < 2 || nAttributes < 1 )
        throw std::string("CalcEuclideanDistanceCoefficient needs data");

    // initialize resemblanceMatrix
    for ( int i = 0; i < nObjects*nObjects; i++ )
        resemblanceMatrix->data.fl[i] = (obj->IsDisimilarityCoeffcient() ? std::numeric_limits<double>::max()
                                         : -1 * std::numeric_limits<double>::max());

    // do the calculations for the each unique combination of objects
    int row_it = 1;
    while ( row_it < nObjects )
    {
        for ( int col_it = 0; col_it < row_it; col_it++ )  // only do calcualation for one direction
        {
            // Euclidean Distance from object at row_it to object at col_it
            double total = 0.0;
            for ( int col = 0; col < nAttributes; col++ )
            {
                double dis =  dataMatrix->data.fl[row_it*nAttributes+col] - \
                            dataMatrix->data.fl[col_it*nAttributes+col];
                total += dis * dis;
            }
            resemblanceMatrix->data.fl[row_it*nObjects+col_it] = sqrt(total);
        }

        row_it++;
    }

}



///////////////////////////////////////////////////////////////////////////
//////////////////// Mahalanobis //////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////

template <typename T>
void CalcMahalanobisDistanceCoefficient(T* obj, int nObjects)
{
    CvMat* dataMatrix = obj->GetDataMatrix();
    CvMat* resemblanceMatrix = obj->GetResemblanceMatrix();
    int nAttributes = obj->GetnAttributes();

    if ( !dataMatrix || !resemblanceMatrix || nObjects < 2 || nAttributes < 1 )
        throw std::string("CalcMahalanobisDistanceCoefficient needs data");

    // initialize resemblanceMatrix
    for ( int i = 0; i < nObjects*nObjects; i++ )
        resemblanceMatrix->data.fl[i] = (obj->IsDisimilarityCoeffcient() ? std::numeric_limits<double>::max()
                                         : -1 * std::numeric_limits<double>::max());

    // do the calculations for the each unique combination of objects
    int row_it = 1;
    while ( row_it < nObjects )
    {
        for ( int col_it = 0; col_it < row_it; col_it++ )  // only do calcualation for one direction
        {
            // Mahalanobis Distance from object at row_it to object at col_it
            double total = 0.0;
            for ( int col = 0; col < nAttributes; col++ )
            {
                double dis =  dataMatrix->data.fl[row_it*nAttributes+col] - \
                            dataMatrix->data.fl[col_it*nAttributes+col];
                total += dis * dis / eigenValueMatrix->data.fl[col_it];
            }
            resemblanceMatrix->data.fl[row_it*nObjects+col_it] = sqrt(total);
        }

        row_it++;
    }
}


////////////////////////////////////////////////////////////////////////////
///////////////////////////////// Revise matrix UPGMA //////////////////////////////////
////////////////////////////////////////////////////////////////////////////



template <typename T>
void ReviseUPGMACoefficientMatrix(T* obj, int nNewObjects, CvMat* newResemblanceMatrix, int removeRow, int removeCol)
{
    CvMat* oldResemblanceMatrix = obj->GetResemblanceMatrix();
    CvMat* originalResemblanceMatrix = obj->GetOriginalResemblanceMatrix();
    int nOldObjects = nNewObjects+1;
    int nOriginalObjects = obj->GetnObjects();
    ClusterContainer& resemblanceLables = obj->GetResemblanceLables();

    for ( int i = 0; i < nNewObjects*nNewObjects; i++ )
        newResemblanceMatrix->data.fl[i] = (obj->IsDisimilarityCoeffcient() ? std::numeric_limits<double>::max()
                                         : -1 * std::numeric_limits<double>::max());

    // iterate through old matrix and pick out the values that we still want to keep
    // simply copy them to the new matrix  // TODO - move this to a different function
    int newrow = 0;
    for ( int oldrow = 0; oldrow < nNewObjects+1; oldrow++ )
    {
        int newcol = 0;
        if ( oldrow != removeRow && oldrow != removeCol )
        {
            for ( int col = 0; col < nNewObjects+1; col++ )
            {
                if ( col != removeRow && col != removeCol )
                {
                    newResemblanceMatrix->data.fl[newrow*nNewObjects+newcol] =
                        oldResemblanceMatrix->data.fl[oldrow*(nOldObjects)+col];
                    newcol++;
                }
            }
            newrow++;
        }
    }

    // the last cluster in the resemblanceLables should be the row we need
    // to find values for
    const Cluster lable1 = resemblanceLables[resemblanceLables.size()-1];
    for ( size_t lable_it = 0; lable_it < resemblanceLables.size()-1; lable_it++ )
    {
        int row = resemblanceLables.size()-1;
        int col = lable_it;
        Cluster lable2 = resemblanceLables[lable_it];
        int count = lable1.objects.size() * lable2.objects.size();

        double newval = 0.0;
        for ( size_t i = 0; i < lable1.objects.size(); i++ )
        {
            for ( size_t j = 0; j < lable2.objects.size(); j++ )
            {
                int oldrow = lable1.objects[i];
                int oldcol = lable2.objects[j];
                if ( oldrow <= oldcol )  // row has to be greater than col
                {
                    int temp = oldrow;
                    oldrow = oldcol;
                    oldcol = temp;
                }
                double valuetoadd = originalResemblanceMatrix->data.fl[oldrow*nOriginalObjects+oldcol];
                newval += valuetoadd;
            }
        }
        newval = newval / (double)count;
        newResemblanceMatrix->data.fl[row*nNewObjects+col] = newval;
    }
}





////////////////////////////////////////////////////////////////////////////
///////////////////////////////// HELPERS //////////////////////////////////
////////////////////////////////////////////////////////////////////////////

template <typename T>
void PrintResemblanceMatrix(T* obj, int nObjects)
{
    CvMat* resemblanceMatrix = obj->GetResemblanceMatrix();

    for ( int i = 0; i < nObjects; i++ )
    {
        for ( int j = 0; j < nObjects-1; j++ )
            std::cout << resemblanceMatrix->data.fl[i*nObjects+j] << ", ";
        std::cout << resemblanceMatrix->data.fl[i*nObjects+(nObjects-1)] << std::endl;
    }
}


/*
    return the min or max value depending on the type of
    Resemblance coefficient in the obj.
    Note: If the resemblance coefficient is a dissimilarity coefficient
    then we return the min value and object index's since they are the most similar,
    if it is a similarity coefficient, than return the objects with the highest value
    in the resemblance matrix.
*/

template <typename T>
void GetResemblanceValue(T* obj, int& object1, int& object2, double& value )
{
    CvMat* resemblanceMatrix = obj->GetResemblanceMatrix();
    CvPoint minloc, maxloc;
    double minval;
    double maxval;
    cvMinMaxLoc(resemblanceMatrix, &minval, &maxval, &minloc, &maxloc);
    if ( obj->IsDisimilarityCoeffcient() )
    {
        value = minval;
        object1 = minloc.y;  // row
        object2 = minloc.x;  // col
    }
    else
    {
        value = maxval;
        object1 = maxloc.x;
        object2 = maxloc.y;
    }
}



template <typename T>
double GetAverageResemblance(T* obj, int nObjects)
{
    CvMat* resemblanceMatrix = obj->GetResemblanceMatrix();
    double avg = 0.0;
    double numerator = 0.0;

    int row_it = 1;
    while ( row_it < nObjects )
    {
        for ( int col_it = 0; col_it < row_it; col_it++ )  // only do calcualation for one direction
        {
            avg += resemblanceMatrix->data.fl[row_it*nObjects+col_it];
            numerator += 1.0;
        }
        row_it++;
    }

    return avg / numerator;
}



// not really related to ResemblanceCoefficents but
// a helpful method
template <typename T>
void PrintDataMatrix(T* obj, int nObjects)
{
    CvMat* dataMatrix = obj->GetDataMatrix();
    int nAttributes = obj->GetnAttributes();

    for ( int i = 0; i < nObjects; i++ )
    {
        for ( int j = 0; j < nAttributes-1; j++ )
            std::cout << dataMatrix->data.fl[i*nAttributes+j] << ", ";
            //printf("%5f, ", dataMatrix->data.fl[i*nObjects+j]);
        std::cout << dataMatrix->data.fl[i*nAttributes+(nAttributes-1)] << std::endl;
        //printf("%5f", dataMatrix->data.fl[i*nObjects+(nObjects-1)]);
    }
}












#endif
