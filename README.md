# Weighted Random Sampling - Alias Tables on the GPU 

This is the code to accompany our technical report "Weighted Random Sampling on GPUs". The report is freely accessible under a Creative Commons license (CC-BY) on [arXiv](https://arxiv.org/abs/2106.12270).

```
@misc{lehmann2021weighted,
    title =   {Weighted Random Sampling on GPUs}, 
    author =  {Hans-Peter Lehmann and Lorenz H{\"u}bschle-Schneider and Peter Sanders},
    year =    {2021},
    journal = {arXiv e-prints},
    eprint =  {2106.12270}
}
```

### Compiling

This project requires CUDA 11.0 or newer. To compile using cmake `cmake`, type the following:

```
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j4
```

### Experiments

To reproduce our experiments, compile using `cmake`. To enable/disable individual experiments, edit the file `plots/Plots.cuh`. Note that some of the experiments (that do not depend on timings) require to use debug builds. Experiments automatically write the output in csv format to both `stdout` and a file. You can specify the file name by changing `OUT_PATH` in `utils/Utils.cuh`.

- You can override some of the hardware-dependent parameters like group size by passing arguments to cmake. See `CMakeLists.txt` for available options.
- To run the experiments where the `l` and `h` arrays do not contain weights, you need to comment out the `LH_TYPE_USE_WEIGHT` definition in `construction/aliasTable/buildMethod/LhType.cuh`.

### License

This code is licensed under the [GPLv3](/LICENSE).
