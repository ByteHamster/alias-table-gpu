#include <iostream>
#include <string>

#include "plots/PlotAdder.cuh"
#include "plots/PlotConstruction.cuh"
#include "plots/PlotConstructionProfile.cuh"
#include "plots/PlotOthers.cuh"
#include "plots/PlotSampling.cuh"
#include "plots/PlotTables.cuh"

#ifndef ALIAS_GPU_PLOT
#define ALIAS_GPU_PLOT
namespace Plot {
    void plotMeasurements() {
        //FILEOUT("addMethods.csv", PlotAdder::plotAddMethods())
        //FILEOUT("prefixSumSpeed.csv", PlotOthers::plotPrefixSumSpeed())
        //FILEOUT("prngSpeed.csv", PlotOthers::plotPrngSpeed())

        //FILEOUT("buildSpeedSplitVariants.csv", PlotConstruction::plotBuildSpeedSplitVariants())
        //FILEOUT("buildSpeedPackVariants.csv", PlotConstruction::plotBuildSpeedPackVariants())
        //FILEOUT("numThreads.csv", PlotConstruction::plotThreadNumber())
        //FILEOUT("numThreadsWeightVsNoWeight.csv", PlotConstruction::plotThreadNumberWeightVsNoWeight())
        //FILEOUT("rejectionSamplingDirectVsBinSearch.csv", PlotConstruction::plotRejectionPerformance())
        //FILEOUT("buildSpeedStructOfArrays.csv", PlotConstruction::plotBuildSpeedStructOfArrays())
        //FILEOUT("buildSpeedWeightDistribution.csv", PlotConstruction::plotBuildSpeedWeightDistribution())
        //FILEOUT("buildSpeedWithTransfer.csv", PlotConstruction::plotBuildSpeedWithTransfer())

        //FILEOUT("buildSpeedLightVsHeavyRandom.csv", PlotConstruction::plotBuildSpeedLightVsHeavy(true))
        //FILEOUT("buildSpeedLightVsHeavyLinear.csv", PlotConstruction::plotBuildSpeedLightVsHeavy(false))
        //FILEOUT("profileLhTypes.csv", PlotConstructionProfile::lhTypes())
        //FILEOUT("profileSharedVsChunked.csv", PlotConstructionProfile::basicVsChunked())
        //FILEOUT("profilePsaPlusUniform.csv", PlotConstructionProfile::psaPlus(weightDistributionUniform))
        //FILEOUT("profilePsaPlusPowerLaw.csv", PlotConstructionProfile::psaPlus(weightDistributionPowerLaw1))
        //FILEOUT("profilePsaPlusPowerLawShuffled.csv", PlotConstructionProfile::psaPlus(weightDistributionPowerLaw05Shuffled))
        //FILEOUT("profileOverallProgress.csv", PlotConstructionProfile::overallProgress())

        //#ifndef LH_TYPE_USE_WEIGHT
        //FILEOUT("profileBasicVsShared.csv", PlotConstructionProfile::basicVsShared())
        //FILEOUT("profileOverallProgress_lh.csv", PlotConstructionProfile::overallProgress())
        //FILEOUT("profileLhTypes_lh.csv", PlotConstructionProfile::lhTypes())
        //#endif

        //#ifndef NDEBUG
        //FILEOUT("samplingDistributionTest.csv", PlotSampling::plotSamplingDistribution(1e5))
        //#endif
        //FILEOUT("samplesPerSecond.csv", PlotSampling::plotSamplingSpeed())
        //FILEOUT("samplesPerSecondStructOfArrays.csv", PlotSampling::plotSamplingSpeedStructOfArrays())
        //FILEOUT("samplesPerSecondWeightDistributionArrayOfStructs.csv", PlotSampling::plotSamplingSpeedWeightDistribution<ArrayOfStructs>())
        //FILEOUT("samplesPerSecondWeightDistributionStructOfArrays.csv", PlotSampling::plotSamplingSpeedWeightDistribution<StructOfArrays>())
        //FILEOUT("samplesPerSecondWeightDistributionRejection.csv", PlotSampling::plotSamplingSpeedWeightDistributionRejection())
        //FILEOUT("samplesPerSecondNumberOfSamples.csv", PlotSampling::plotNumberOfSamples())
        //FILEOUT("samplesPerSecondSamplerMethodDetail1e6.csv", PlotSampling::plotSamplingSpeedSamplerMethod(1e6, 6e7, 4e5))
        //FILEOUT("samplesPerSecondSamplerMethodMax1e6.csv", PlotSampling::plotSamplingSpeedSamplerMethod(1e6, 5e8, 1e7))
        //FILEOUT("samplesPerSecondSamplerMethodDetail1e7.csv", PlotSampling::plotSamplingSpeedSamplerMethod(1e7, 8e7, 1e6))
        //FILEOUT("samplesPerSecondSamplerMethodMax1e7.csv", PlotSampling::plotSamplingSpeedSamplerMethod(1e7, 8e8, 1e7))
        //FILEOUT("samplesPerSecondThrashing.csv", PlotSampling::plotSamplingSpeedThrashing(5e8, 5e6))
        //FILEOUT("samplesPerSecondMethodHeatmap.csv", PlotSampling::plotSamplingSpeedMethodHeatmap())
        //FILEOUT("samplesPerSecondItemsPerSection1e6.csv", PlotSampling::plotSamplingSpeedSectionedItemsPerSection(1e6, 12000))
        //FILEOUT("samplesPerSecondItemsPerSection1e7.csv", PlotSampling::plotSamplingSpeedSectionedItemsPerSection(1e7, 6000))
        //FILEOUT("samplesPerSecondGroupSizeXorWow.csv", PlotSampling::plotSamplerGroupSize<XorWowPrng>(1024))
        //FILEOUT("samplesPerSecondGroupSizeMt.csv", PlotSampling::plotSamplerGroupSize<MtPrng>(256))
        //FILEOUT("samplesPerSecondDirectComparison1e6.csv", PlotSampling::directMethodComparison(1e6))
        //FILEOUT("samplesPerSecondDirectComparison1e7.csv", PlotSampling::directMethodComparison(1e7))

        //#ifndef SET_CACHE_CONFIG
        //FILEOUT("samplesPerSecondCacheConfig.csv", PlotSampling::cacheConfig())
        //#endif

        //PlotTables::comparisonLhs();
        //PlotTables::basicSamplerComparison();
        //PlotTables::basicBadPackMethods();
        //#ifndef LH_TYPE_USE_WEIGHT
        //PlotTables::basicBadPackMethodPrecomputedWeight();
        //#endif

        //#ifndef NDEBUG
        //FILEOUT("ks/1e5.csv", PlotTables::plotKsTest(1e6, 1e5))
        //FILEOUT("ks/1e6.csv", PlotTables::plotKsTest(1e6, 1e6))
        //FILEOUT("ks/1e7.csv", PlotTables::plotKsTest(1e6, 1e7))
        //#endif
    }
}
#endif // ALIAS_GPU_PLOT
