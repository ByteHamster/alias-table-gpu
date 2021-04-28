#ifndef ALIAS_GPU_TABLESTORAGE_CUH
#define ALIAS_GPU_TABLESTORAGE_CUH

#include "utils/SafeArray.cuh"

class ArrayOfStructs {
    public:
        struct TableRow {
            double weight;
            int alias;
        };

        SafeArray<TableRow> rows;
        int size = 0;

        explicit ArrayOfStructs(StorageLocation _location) : rows(_location) {
        }

        explicit ArrayOfStructs(size_t n, StorageLocation _location) : ArrayOfStructs(_location) {
            malloc(n);
        }

        __host__ __device__ __forceinline__
        double &weight(int index) {
            return rows[index].weight;
        }

        __host__ __device__ __forceinline__
        int &alias(int index) {
            return rows[index].alias;
        }

        __host__ __device__ __forceinline__
        void setBoth(int index, ArrayOfStructs::TableRow both) {
            ASSIGN_128(rows.data[index], both)
            //rows[index] = both;
        }

        __host__ __device__ __forceinline__
        ArrayOfStructs::TableRow getBoth(int index) {
            ArrayOfStructs::TableRow row = {};
            ASSIGN_128(row, rows.data[index])
            return row;
        }

        void malloc(size_t n) {
            size = n;
            rows.malloc(n);
        }

        void free() {
            size = 0;
            rows.free();
        }

        void copyFrom(ArrayOfStructs other) {
            rows.copyFrom(other.rows);
        }

        void print(std::string title, int maxNum) {
            rows.print(title, maxNum);
        }

        static std::string name() {
            return "ArrayOfStructs";
        }
};

class StructOfArrays {
    public:
        SafeArray<double> weights;
        SafeArray<int> aliases;
        int size = 0;

        explicit StructOfArrays(StorageLocation _location) : weights(_location), aliases(_location) {
        }

        explicit StructOfArrays(size_t n, StorageLocation _location) : StructOfArrays(_location) {
            malloc(n);
        }

        __host__ __device__ __forceinline__
        double &weight(int index) {
            return weights[index];
        }

        __host__ __device__ __forceinline__
        int &alias(int index) {
            return aliases[index];
        }

        __host__ __device__ __forceinline__
        void setBoth(int index, ArrayOfStructs::TableRow both) {
            weights[index] = both.weight;
            aliases[index] = both.alias;
        }

        __host__ __device__ __forceinline__
        ArrayOfStructs::TableRow getBoth(int index) {
            return {weights[index], aliases[index]};
        }

        void malloc(size_t n) {
            size = n;
            weights.malloc(n);
            aliases.malloc(n);
        }

        void free() {
            size = 0;
            weights.free();
            aliases.free();
        }

        void copyFrom(StructOfArrays other) {
            weights.copyFrom(other.weights);
            aliases.copyFrom(other.aliases);
        }

        void print(std::string title, int maxNum) {
            weights.print(title + std::string("-weights"), maxNum);
            aliases.print(title + std::string("-aliases"), maxNum);
        }

        static std::string name() {
            return "StructOfArrays";
        }
};

inline std::ostream& operator<<(std::ostream &os, ArrayOfStructs::TableRow &row) {
    os << "(" << row.weight << ", " << row.alias << ")";
    return os;
}

#endif //ALIAS_GPU_TABLESTORAGE_CUH
