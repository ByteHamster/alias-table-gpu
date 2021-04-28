#include <iostream>
#include <string>

#include "construction/aliasTable/buildMethod/AliasTableSplitGpu.cuh"

#ifndef ALIAS_GPU_PLOT_CONSTRUCTION_PROFILE
#define ALIAS_GPU_PLOT_CONSTRUCTION_PROFILE

namespace PlotConstructionProfile {

    TableSplitTimer::TimingResult profile(
            AliasTableSplitGpu<ArrayOfStructs> &tableSplitGpu, int iterations) {
        tableSplitGpu.preBuild();
        TableSplitTimer::TimingResult timing;
        for (int i = 0; i < iterations; i++) {
            tableSplitGpu.build();
            timing = timing + tableSplitGpu.timer.get();
        }
        tableSplitGpu.postBuild();
        timing = timing / iterations;
        return timing;
    }

    void profileVariants(int n, WeightDistribution weightDistribution,
                         Variant variants[], int numVariants, std::string additional) {
        for (int i = 0; i < numVariants; i++) {
            AliasTableSplitGpu<ArrayOfStructs> tableSplitGpu(n, variants[i], NUMBER_THREADS_AUTO, weightDistribution);
            TableSplitTimer::TimingResult timing = PlotConstructionProfile::profile(tableSplitGpu, 500);
            std::cout<<timing.print(tableSplitGpu.name()
                    + (additional.length() == 0 ? "" : ";" + additional))<<std::endl;
        }
    }

    void basicVsShared() {
        #ifdef LH_TYPE_USE_WEIGHT
            std::cerr<<"Test created before weights were added to LH"<<std::endl;
            raise(SIGABRT);
        #endif
        const int numVariants = 2;
        Variant variants[numVariants] = {
                Variant(splitVariantParySearch, packVariantBasic, false),
                Variant(splitVariantParySearch, packVariantBasic, true)
        };
        std::cout<<"step;duration;variant"<<std::endl;
        profileVariants(1e7, weightDistributionUniform, variants, numVariants, "");
    }

    void lhTypes() {
        const int numVariants = 1;
        Variant variants[numVariants] = {
                Variant(splitVariantParySearch, packVariantBasic, true)
        };
        std::cout<<"step;duration;variant;lhType"<<std::endl;
        profileVariants(1e7, weightDistributionUniform, variants, numVariants, LH_TYPE::name());
    }

    void basicVsChunked() {
        const int numVariants = 2;
        Variant variants[numVariants] = {
                Variant(splitVariantParySearch, packVariantBasic, true),
                Variant(splitVariantParySearch, packVariantChunkedShared, false)
        };
        std::cout<<"step;duration;variant"<<std::endl;
        profileVariants(1e7, weightDistributionUniform, variants, numVariants, "");
    }

    void psaPlus(WeightDistribution weightDistribution) {
        const int numVariants = 2;
        Variant variants[numVariants] = {
                Variant(splitVariantParySearch, packVariantChunkedShared, true),
                Variant(splitVariantParySearch, packVariantChunkedShared, true)
        };
        variants[1].setPsaPlusIfSupported(true);
        std::cout<<"step;duration;variant"<<std::endl;
        profileVariants(1e7, weightDistribution, variants, numVariants, "");
    }

    void overallProgress() {
        #ifndef LH_TYPE_USE_WEIGHT
            const int numVariants = 3;
            Variant variants[numVariants] = {
                    Variant(splitVariantBasic, packVariantBasic, false), // devicePartition off
                    Variant(splitVariantBasic, packVariantBasic, false),
                    Variant(splitVariantBasic, packVariantBasic, true)
            };
            variants[0].devicePartition = false;
        #else
            const int numVariants = 4;
            Variant variants[numVariants] = {
                    Variant(splitVariantBasic, packVariantBasic, true),
                    Variant(splitVariantParySearch, packVariantBasic, true),
                    Variant(splitVariantParySearch, packVariantChunkedShared, false),
                    Variant(splitVariantParySearch, packVariantChunkedShared, false) //PSA+
            };
            variants[numVariants - 1].setPsaPlusIfSupported(true);
        #endif
        std::cout<<"step;duration;variant;lhType"<<std::endl;
        profileVariants(1e7, weightDistributionUniform, variants, numVariants, LH_TYPE::name());
    }
}
#endif // ALIAS_GPU_PLOT_CONSTRUCTION_PROFILE
