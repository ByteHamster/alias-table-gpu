#include <ctime>

#include "plots/Plot.cuh"
#include "UnitTests.cuh"

int main() {
    srand(static_cast <unsigned> (time(nullptr)));
    UnitTests::execute();
    Plot::plotMeasurements();

    ///////////////////// Playground /////////////////////

    Variant variants[] = {
        Variant(splitVariantBasic, packVariantBasic, false),
        Variant(splitVariantParySearch, packVariantBasic, true),
        Variant(splitVariantParySearch, packVariantChunkedShared, false),
    };

    for (Variant variant : variants) {
        AliasTableSplitGpu<ArrayOfStructs> tableSplitGpu(1e7, variant,
                 NUMBER_THREADS_AUTO, weightDistributionPowerLaw1Shuffled);
        PlotConstruction::plotBuildSpeed(tableSplitGpu, "", 100);
    }
    return 0;
}
