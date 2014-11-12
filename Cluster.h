#ifndef CLUSTER_H
#define CLUSTER_H

struct Cluster
{
    std::vector<int> objects;
    double           distance;
    bool             bIsNew;

    bool operator!=(const Cluster &rhs)
    {
        for ( size_t i = 0; i < objects.size(); i++ )
        {
            int comp = objects[i];
            for ( size_t j = 0; j < rhs.objects.size(); j ++ )
                if ( comp == rhs.objects[j] )
                    return false;
        }
        return true;
    }

};

typedef std::vector<Cluster> ClusterContainer;


struct Cluster_step
{
    ClusterContainer clusters;
};




#endif
