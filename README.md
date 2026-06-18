### Lanczos image resizing

- C++20, tbb library a parallelizalashoz
- stb_image es stb_image_write headerok a kep olvasas/irashoz
- CUDA a GPU implementaciohoz

```
OS: Ubuntu 26.04 LTS (kernel 7.0.0-22-generic, x86_64)
g++ (Ubuntu 15.2.0-16ubuntu1) 15.2.0
Cuda compilation tools, release 12.4, V12.4.131
NVIDIA GeForce RTX 2060 SUPER, 7.5, 595.71.05
Intel oneTBB: 2022.3 (libtbb12 2022.3.0-2)
```

#### Build

CPU-only binary (`lanczos`):

```
g++ -std=c++20 -O3 -march=native -ffp-contract=off lanczos.cpp -o lanczos -ltbb -pthread
```

CPU+GPU comparison binary (`lanczos_compare`):

```
g++ -std=c++20 -O3 -march=native -ffp-contract=off -DLANCZOS_NO_MAIN -DLANCZOS_NO_STB_IMPL -c lanczos.cpp -o lanczos.cpp.o
nvcc -std=c++20 -O3 --fmad=false -c lanczos.cu -o lanczos.cu.o
g++ lanczos.cpp.o lanczos.cu.o -o lanczos_compare -L/usr/local/cuda/lib64 -lcudart -ltbb -pthread
```

- `fmad=false` es `ffp-contract=off`, kulonben kerekites miatt nem jonnek ki a tesztek


#### CLI

```
./lanczos <input> <output> <new_width> <new_height> [a]
./lanczos_compare --cpu|--gpu|--compare <input> <output> <new_width> <new_height> [a]
./lanczos_compare --bench <input> <output> <new_width> <new_height> <a> <runs>
./lanczos_compare --test
```

- alapertelmezett a = 3
- `--bench` lefuttatja 'runs'-szor, visszaadja a min/median/max futasi idot
- `--test` testeket futtat: konstans, identity, random

Pelda:

- `./lanczos fig_spectrum.png fig_res.png 2000 2000 3`
- `./lanczos_compare --compare white.jpg white_res.jpg 1000 1000 3`
- `./lanczos_compare --bench white.jpg out.jpg 1000 1000 3 30`
- `./lanczos_compare --test`


