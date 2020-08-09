# Mind's Crawl

[![Codacy Badge](https://api.codacy.com/project/badge/Grade/cd8ac0eb5b334c68b7661ab756049191)](https://app.codacy.com/gh/physarumAdv/mind-s_crawl?utm_source=github.com&utm_medium=referral&utm_content=physarumAdv/mind-s_crawl&utm_campaign=Badge_Grade_Dashboard)

This is a simulation of Physarum Polycephalum, written in CUDA, but can be compiled for CPU execution with NVCC.
At the moment, the code is very raw, it is going to be better commented and rearranged in files

### Compiling

**PLEASE, NOTE** that there is no default `CMakeLists.txt` file, so you have to symbolically **link** `CMakeLists.txt`
to either `cmakelists_parts/cpu/CMakeLists.txt` or `cmakelists_parts/gpu/CMakeLists.txt`, for example:

```bash
cd project_dir
ln -s cmakelists_parts/gpu/CMakeLists.txt CMakeLists.txt
```

After that, you will be able to build the project. The executable will run on either CPU or GPU depending on
which file you choose

## Authors

> [Nikolay Nechaev](http://t.me/kolayne), [nikolay_nechaev@mail.ru](mailto:nikolay_nechaev@mail.ru)
>
> [Tatiana Kadykova](http://vk.com/ricopin), [tanya-kta@bk.ru](mailto:tanya-kta@bk.ru)
>
> [Pavel Artushkov](http://t.me/pavtiger), [pavTiger@gmail.com](mailto:pavTiger@gmail.com)
>
> [Olga Starunova](http://vk.com/id2051067), [bogadelenka@mail.ru](mailto:bogadelenka@mail.ru)
