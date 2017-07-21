
## Installation Instructions

This project mostly requires TensorFlow and his dependencies :

- TensorFlow (>1.x.y) + TensorBoard
- Numpy
- scipy
- scikit-learn & scikit-image
- matplotlib
- jupyter notebooks

You can easily install thoses libraries manually, or use :
- conda
- docker & nvidia-docker


### with Conda

#### Install Miniconda

If Miniconda or Conda is not installed :
_Note_ : here for Linux 64-bit systems.

```bash
wget https://repo.continuum.io/miniconda/Miniconda3-4.2.12-Linux-x86_64.sh
bash Miniconda3-4.2.12-Linux-x86_64.sh -b
rm Miniconda3-4.2.12-Linux-x86_64.sh
```

Don't forget to add `~/miniconda3/bin/` to your `$PATH` (depending on your PREFIX during Miniconda installation).

#### Create a new env

Dependencies are detailed in `environment.yml` and `environment-gpu.yml`.
Those commands create a new env, named `dl-st`

```bash
cd install/

# CPU
conda env create -f=environment.yml --name dl-st --debug -v -v

# GPU
# requires GPU supports : Nvidia drivers, CUDA, cuDNN, ...
conda env create -f=environment-gpu.yml --name dl-st --debug -v -v
```

For GPU support, don't forget to set up correctly environment variables (CUDA_PATH, LD_LIBRARY_PATH, ...).

Now you can activate/deactivate it with :

```bash
source activate dl-st

# run jupyter :
jupyter notebook [specific notebook] &
tensorboard --logdir=logs/ &

source deactivate dl-st

```



### with `docker`

If you are familiar with docker, I wrote two dockerfiles.

Images are based on `tensorflow/tensorflow:1.2.0-py3` or `tensorflow/tensorflow:1.2.0-gpu-py3`.

Additional dependencies are added with `pip` during the building step.

You can find instructions for docker installation [here](http://docs.docker.com/engine/installation/).

For GPU support, I use `nvidia-docker`. You can find more information [here](http://github.com/NVIDIA/nvidia-docker).



#### CPU support

```bash
cd install/

# build image
docker build -t dl-st:cpu -f Dockerfile.cpu .

cd ../

# run a container from the root directory : it starts automatically jupyter
docker run -it -p 8888:8888 -p 6666:6666 -v "$PWD"/:/notebooks/ dl-st:cpu

# if you want command line acces
docker run -it -p 8888:8888 -p 6666:6666 -v "$PWD"/:/notebooks/ dl-st:cpu bash

(root@...) cd notebooks/
(root@...) jupyter notebook --allow-root &
(root@...) tensorboard logdir=logs/ &

```



#### GPU support with `nvidia-docker`


```bash
cd install/

# build image
docker build -t dl-st:gpu -f Dockerfile.gpu .

cd ../

# run a container from the root directory : it starts automatically jupyter
nvidia-docker run -it -p 8888:8888 -p 6006:6006 -v "$PWD"/:/notebooks/ dl-st:gpu

# if you want command line acces
docker run -it -p 8888:8888 -p 6006:6006 -v "$PWD"/:/notebooks/ dl-st:cpu bash

(root@...) cd notebooks/
(root@...) jupyter notebook --allow-root &
(root@...) tensorboard logdir=logs/ &

```


#### Accessing Jupyter notebook and TensorBoard :

When you run `docker run -it -p 8888:8888 -p 6666:6666 -v "$PWD"/:/notebooks/ dl-st:cpu` it prints the token for the Jupyter server.
Open this link to get access to the notebooks.

Tensorboard is available at `https://0.0.0.0:6006`.
TensorBoard is launched with `--logdir=logs/`, so you have to manually select the appropriate run in TensorBoard.

To kill the container : `Ctrl+C`.