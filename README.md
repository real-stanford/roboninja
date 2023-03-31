# RoboNinja: Learning an Adaptive Cutting Policy for Multi-Material Objects


[Zhenjia Xu](http://www.zhenjiaxu.com/)<sup>1</sup>,
[Zhou Xian](https://www.zhou-xian.com/)<sup>2</sup>,
[Xingyu Lin](https://xingyu-lin.github.io/)<sup>3</sup>,
[Cheng Chi](https://cheng-chi.github.io/)<sup>1</sup>,
[Zhiao Huang](https://sites.google.com/view/zhiao-huang)<sup>4</sup>,
[Chuang Gan](https://people.csail.mit.edu/ganchuang/)<sup>5&dagger;</sup>,
[Shuran Song](https://www.cs.columbia.edu/~shurans/)<sup>1&dagger;</sup>
<br>
<sup>1</sup>Columbia University, <sup>2</sup>CMU, <sup>3</sup>UC Berkeley, <sup>4</sup>UC San Diego, <sup>5</sup>UMass Amherst & MIT-IBM Lab

### [Project Page](https://roboninja.cs.columbia.edu/) | [Video](https://youtu.be/SyEAP_jlgSQ) | [arXiv](https://arxiv.org/abs/2302.11553)


## Installation
We recommend [Mambaforge](https://github.com/conda-forge/miniforge#mambaforge) instead of the standard anaconda distribution for faster installation: 
```console
$ mamba env create -f environment.yml
```

but you can use conda as well: 
```console
$ conda env create -f environment.yml
```

## Simulation
Generate cores with in-distribution geometries (300 train + 50 eval)
```console
$ python roboninja/workspace/bone_generation_workspace.py
```
Generate cores with out-of-distribution geometries (50 eval)
```console
$ python roboninja/workspace/bone_generation_ood_workspace.py
```
[simulation_example.ipynb](simulation_example.ipynb) provides a quick example of the simulation. It first create a scene and render an image. It then runs a forward pass a backward pass using the initial action trajectory. Finally, it executes an optimized action trajectory.

If you get an error related to rendering, here are some potential solutions:
- make sure [vulkan](https://www.vulkan.org/) is installed
- `TI_VISIBLE_DEVICE` is not correctly set in [roboninja/env/tc_env.py](roboninja/env/tc_env.py) (L17). The reason is that vukan device index is not alighed with cuda device index, and that's the reason I have the function called `get_vulkan_offset()`. Change this function implementation based on your setup.