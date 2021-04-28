#include "PackMethodSweep.cuh"

template <typename TableStorage>
__host__ __device__
void PackMethodSweep::pack(int k, SafeArray<SplitConfig> splits, double W_N,
        SafeArray<LH_TYPE> h, SafeArray<LH_TYPE> l, TableStorage aliasTable, SafeArray<double> weights) {
    SplitConfig splitCurrent = splits[k];
    SplitConfig splitPrevious = splits[k - 1];
    int iLower = splitPrevious.i;
    int iUpper = splitCurrent.i;
    int jLower = splitPrevious.j;
    int jUpper = splitCurrent.j;
    double spill = splitPrevious.spill;

    int i = l[iLower].item;
    int j = h[jLower].item;
    int iMax = l[iUpper].item; // Actual max item index, no l/h indirection
    int jMax = h[jUpper].item;

    double w = spill;
    if (spill == 0) {
        w = h[jLower].getWeight(weights.data);
    }

    //std::cout<<"START: "<<"i="<<i<<" j="<<j<<" w="<<w<<std::endl;
    while (true) {
        bool weightExhausted = w <= W_N + EPSILON;
        if (weightExhausted && j >= jMax) {
            weightExhausted = false;
        } else if (!weightExhausted && i >= iMax) {
            weightExhausted = true;
        }
        if (weightExhausted) {
            if (j >= jMax) {
                //std::cout<<"END: "<<"i="<<i<<" j="<<j<<" w="<<w<<std::endl;
                assert(i == iMax);
                return;
            }

            int j_ = j;
            do {
                j_++;
            } while (j < jMax && weights[j_] <= W_N);

            aliasTable.weight(j) = w;
            aliasTable.alias(j) = j_;

            w += weights[j_] - W_N;
            j = j_;
        } else {
            if (i >= iMax) {
                //std::cout<<"END: "<<"i="<<i<<" j="<<j<<" w="<<w<<std::endl;
                assert(j == jMax);
                return;
            }
            aliasTable.weight(i) = weights[i];
            aliasTable.alias(i) = j;
            w += aliasTable.weight(i) - W_N;

            do {
                i++;
            } while (i < iMax && weights[i] > W_N);
        }
    }
}

template <typename TableStorage>
__host__ __device__
void PackMethodSweep::packOptimized(int k, SafeArray<SplitConfig> splits, double W_N,
        LH_TYPE const *h, LH_TYPE const *l, TableStorage aliasTable, double const *weights) {
    SplitConfig splitCurrent = splits[k];
    SplitConfig splitPrevious = splits[k - 1];
    int iLower = splitPrevious.i;
    int iUpper = splitCurrent.i;
    int jLower = splitPrevious.j;
    int jUpper = splitCurrent.j;
    double spill = splitPrevious.spill;

    int i = l[iLower].item;
    int j = h[jLower].item;
    int iMax = l[iUpper].item; // Actual max item index, no l/h indirection
    int jMax = h[jUpper].item;

    double w = spill;
    if (spill == 0) {
        w = h[jLower].getWeight(weights);
    }

    //std::cout<<"START: "<<"i="<<i<<" j="<<j<<" w="<<w<<std::endl;
    int aliasLocation;
    ArrayOfStructs::TableRow rowTemp = {};
    while (true) {
        bool weightExhausted = w <= W_N + EPSILON;
        if (weightExhausted && j >= jMax) {
            weightExhausted = false;
        } else if (!weightExhausted && i >= iMax) {
            weightExhausted = true;
        }

        int indexUpdate;
        int indexUpdateMax;

        if (weightExhausted) {
            if (j >= jMax) {
                //std::cout<<"END: "<<"i="<<i<<" j="<<j<<" w="<<w<<std::endl;
                assert(i == iMax);
                return;
            }
            indexUpdate = j;
            indexUpdateMax = jMax;
            aliasLocation = j;
        } else {
            if (i >= iMax) {
                //std::cout<<"END: "<<"i="<<i<<" j="<<j<<" w="<<w<<std::endl;
                assert(j == jMax);
                return;
            }
            indexUpdate = i;
            indexUpdateMax = iMax;
            aliasLocation = i;
        }

        do {
            indexUpdate++;

            double weight = weights[indexUpdate];
            if (weightExhausted && weight > W_N) {
                break;
            } else if (!weightExhausted && weight <= W_N) {
                break;
            }
        } while (indexUpdate < indexUpdateMax);

        int nextElementWeightLocation;
        if (weightExhausted) {
            rowTemp.weight = w;
            rowTemp.alias = indexUpdate;
            nextElementWeightLocation = indexUpdate;
            j = indexUpdate;
        } else {
            rowTemp.weight = weights[aliasLocation];
            rowTemp.alias = j;
            nextElementWeightLocation = i;
            i = indexUpdate;
        }

        w += weights[nextElementWeightLocation] - W_N;
        aliasTable.setBoth(aliasLocation, rowTemp);
    }
}

template <typename TableStorage>
__global__
void PackMethodSweep::packKernel(int splitOffset, SafeArray<SplitConfig> splits, double W_N,
              SafeArray<LH_TYPE> h, SafeArray<LH_TYPE> l, TableStorage aliasTable, int p, SafeArray<double> weights) {
    unsigned int k = blockIdx.x * blockDim.x + threadIdx.x + 1 + splitOffset;
    if (k > p) {
        return;
    }
    PackMethodSweep::packOptimized(k, splits, W_N, h.data, l.data, aliasTable, weights.data);
}

template
__host__ __device__
void PackMethodSweep::pack<ArrayOfStructs>(int k, SafeArray<SplitConfig> splits, double W_N,
       SafeArray<LH_TYPE> h, SafeArray<LH_TYPE> l, ArrayOfStructs aliasTable, SafeArray<double> weights);

template
__host__ __device__
void PackMethodSweep::packOptimized<ArrayOfStructs>(int k, SafeArray<SplitConfig> splits, double W_N,
        LH_TYPE const *h, LH_TYPE const *l, ArrayOfStructs aliasTable, double const *weights);

template
__global__
void PackMethodSweep::packKernel<ArrayOfStructs>(int splitOffset, SafeArray<SplitConfig> splits, double W_N,
        SafeArray<LH_TYPE> h, SafeArray<LH_TYPE> l, ArrayOfStructs aliasTable, int p, SafeArray<double> weights);

template
__host__ __device__
void PackMethodSweep::pack<StructOfArrays>(int k, SafeArray<SplitConfig> splits, double W_N,
        SafeArray<LH_TYPE> h, SafeArray<LH_TYPE> l, StructOfArrays aliasTable, SafeArray<double> weights);

template
__host__ __device__
void PackMethodSweep::packOptimized<StructOfArrays>(int k, SafeArray<SplitConfig> splits, double W_N,
        LH_TYPE const *h, LH_TYPE const *l, StructOfArrays aliasTable, double const *weights);

template
__global__
void PackMethodSweep::packKernel<StructOfArrays>(int splitOffset, SafeArray<SplitConfig> splits, double W_N,
        SafeArray<LH_TYPE> h, SafeArray<LH_TYPE> l, StructOfArrays aliasTable, int p, SafeArray<double> weights);
