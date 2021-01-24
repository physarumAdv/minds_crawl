# Mind's Crawl

[![Codacy Badge](https://api.codacy.com/project/badge/Grade/cd8ac0eb5b334c68b7661ab756049191)](https://app.codacy.com/gh/physarumAdv/minds_crawl?utm_source=github.com&utm_medium=referral&utm_content=physarumAdv/minds_crawl&utm_campaign=Badge_Grade_Dashboard)
[![Ubuntu build](https://github.com/physarumAdv/minds_crawl/workflows/Ubuntu%20build/badge.svg)](https://github.com/physarumAdv/minds_crawl/actions?query=workflow%3A%22Ubuntu+build%22)

This is a simulation of Physarum Polycephalum, written in CUDA, but can be compiled for CPU execution with NVCC.

## Compiling

Before compiling the project, clone the repository with the submodules:
```bash
git clone https://github.com/physarumAdv/minds_crawl.git --recursive
cd minds_crawl
```

To compile (the produced executable will require an NVidia GPU to run):
```bash
mkdir cmake-build-release && cd cmake-build-release
cmake ..
cmake --build . -- -j`nproc`
```

Note that there is also a way to produce an executable which will only use CPU for running, however it's highly
unrecommended to use this mode for any purposes but debugging:
```bash
mkdir cmake-build-debug && cd cmake-build-debug
cmake .. -DCMAKE_BUILD_TYPE=Debug -DCOMPILE_FOR_CPU=ON
cmake --build . -- -j`nproc`
```

### Executing
To run the application, you have to:

1.  Start a visualization app (we recommend [this one](https://github.com/physarumAdv/Physarum_webGL))

2.  Create a directory `config` in the process's **working directory** with a file `visualization_endpoint.txt` inside
it, containing 2 urls which accepts the simulated data (One url for particles and one for polyhedron. An example is in
[local/visualization_endpoint_example.txt](config/visualization_endpoint_example.txt))

## Authors

> [Nikolay Nechaev](http://t.me/kolayne), [nikolay_nechaev@mail.ru](mailto:nikolay_nechaev@mail.ru)
>
> [Tatiana Kadykova](http://vk.com/ricopin), [tanya-kta@bk.ru](mailto:tanya-kta@bk.ru)
>
> [Pavel Artushkov](http://t.me/pavtiger), [pavTiger@gmail.com](mailto:pavTiger@gmail.com)
>
> [Olga Starunova](http://vk.com/id2051067), [bogadelenka@mail.ru](mailto:bogadelenka@mail.ru)
