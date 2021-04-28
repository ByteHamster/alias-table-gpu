#ifndef ALIAS_GPU_LHTYPE_CUH
#define ALIAS_GPU_LHTYPE_CUH

#define LH_TYPE_USE_WEIGHT
#ifdef LH_TYPE_USE_WEIGHT
    #define LH_TYPE LhTypeWithWeight
    #define ASSIGN_LH(dest, src) { \
        ASSIGN_128(dest, src) \
    }
#else
    #define LH_TYPE LhTypeNoWeight
    #define ASSIGN_LH(dest, src) { \
        dest = src;\
    }
#endif

struct LhTypeWithWeight {
    int item;
    double weight;

    __host__ __device__ __forceinline__
    double getWeight(double const *weights) const {
        return weight;
    }

    template<typename TableStorage>
    __host__ __device__ __forceinline__
    double getWeightFromTable(TableStorage aliasTable) const {
        return weight;
    }

    __host__ __device__ __forceinline__
    void setWeight(double weight) {
        this->weight = weight;
    }

    static std::string name() {
        return "LhTypeWithWeight";
    }
};

struct LhTypeNoWeight {
    int item;

    __host__ __device__ __forceinline__
    double getWeight(double const *weights) const {
        return weights[item];
    }

    template<typename TableStorage>
    __host__ __device__ __forceinline__
    double getWeightFromTable(TableStorage aliasTable) const {
        return aliasTable.weight(item);
    }

    __host__ __device__ __forceinline__
    void setWeight(double weight) const {
    }

    static std::string name() {
        return "LhTypeNoWeight";
    }
};

inline std::ostream& operator<<(std::ostream &os, LhTypeWithWeight &item) {
    os << "(" << item.item << ", " << item.weight << ")";
    return os;
}

inline std::ostream& operator<<(std::ostream &os, LhTypeNoWeight &item) {
    os << item.item;
    return os;
}

#endif //ALIAS_GPU_LHTYPE_CUH
