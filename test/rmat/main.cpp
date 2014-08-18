#include <cstdio>

#include <vector>
#include <algorithm>

using namespace std;

struct edge {
    long i;
    long j;
};

void kroneker(long SF, std::vector<edge>& edgelist){
    std::default_random_engine generator;
    std::uniform_real_distribution<> distribution(0.,1.);

    // Set initiator probabilities.
    const double A = 0.57;
    const double B = 0.19;
    const double C = 0.19;

    // Loop over each order of bit.
    const double ab = A + B;
    const double c_norm = C/(1 - (A + B));
    const double a_norm = A/(A + B);

    for(auto& out: edgelist){
        long i = 0;
        long j = 0;
        for(long it=0; it<SF; it++){
            // Compare with probabilities and set bits of indices.
            long ii_bit = (distribution(generator) > ab)? 1 : 0;
            long jj_bit = (distribution(generator) > ( c_norm * ii_bit + a_norm * !(ii_bit) ))? 1: 0;
            i += ii_bit << it;
            j += jj_bit << it;
        }
        out = edge{i,j};
    }

}

int main()
{

    //parameter
    const long scalefactor = 8;
    const long edgefactor = 16;

    const long vertices = 1 << scalefactor;
    long numberOfEdges = vertices*edgefactor;

    std::vector<edge> edgelist(numberOfEdges);
    //generate edgelist
    kroneker(scalefactor,edgelist);
    //both directions
    edgelist.resize(2*numberOfEdges);
    for(auto i= 0; i < numberOfEdges; i++){
        edgelist[numberOfEdges+i]=edge{edgelist[i].j,edgelist[i].i};
    }
    //sort list
    std::sort(edgelist.begin(),edgelist.end(),
              [](const edge& i, const edge& j)->bool{
                    return (i.i < j.i) ? true : ((i.i > j.i)? false:  (i.j < j.j));
    });
    //remove duplicates
    auto it_end = std::unique( edgelist.begin(), edgelist.end(), [](const edge& i, const edge& j){return (i.i == j.i && i.j == j.j);});

    for(auto it=edgelist.begin(); it != it_end; it++){
        printf("%ld %ld\n",it->i, it->j);
    }

}

