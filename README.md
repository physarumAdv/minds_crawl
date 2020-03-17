# Mind's Crawl
This is a simulation of Physarum Polycephalum, written in CUDA, but can be compiled for CPU execution with NVCC.
At the moment, the code is very raw, it is going to be better commented and rearranged in files
### Compiling
**PLEASE, NOTE** that there is no default `CMakeLists.txt` file, so you have to symbolically **link** either
`CMakeLists_cpu.txt` or `CMakeLists_gpu.txt` to `CMakeLists.txt`, for example:
```bash
cd project_dir
ln -s CMakeLists_gpu.txt CMakeLists.txt
```
After that, you will be able to build the project. The executable will run on either CPU or GPU depending on
which file you choose
