#include "AliasTableStack.cuh"

std::string AliasTableStack::name() {
    return "AliasTableStack";
}

AliasTableStack::AliasTableStack(int size, WeightDistribution weightDistribution) : AliasTable(size, weightDistribution) {

}

void AliasTableStack::build() {
    sumWeightsCpu();

    for (int i=0; i < N; i++) {
        if (weights[i] > W/N) {
            h.push_back(i);
        } else {
            l.push_back(i);
        }
    }
    assert(!h.empty());
    assert(!l.empty());

    for (int i=0; i < N; i++) {
        aliasTable.weight(i) = weights[i];
    }

    while (!h.empty() && /* Not in the paper: */ !l.empty()) {
        int j = h.back();
        h.pop_back();
        while (aliasTable.weight(j) > (float) W/N && /* Not in the Paper: */ !l.empty()) {
            int i = l.back();
            l.pop_back();
            aliasTable.alias(i) = j;
            aliasTable.weight(j) += aliasTable.weight(i) - W/N;
        }
        l.push_back(j);
    }
}
