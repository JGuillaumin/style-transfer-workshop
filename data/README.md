
## VGG networks

I was inspired by the work of [**machrisaa**](http://github.com/machrisaa/tensorflow-vgg).

The Numpy files are :

- [vgg16.npy](http://mega.nz/#!YU1FWJrA!O1ywiCS2IiOlUCtCpI6HTJOMrneN-Qdv3ywQP5poecM)
- [vgg19.npy](http://mega.nz/#!xZ8glS6J!MAnE91ND_WyfZ_8mvkuSa2YcA7q-1ehfSm-Q1fxOvvs)

These files must be placed into `data/VGG/*.npy`.

Notebooks `0.1_VGG_construction.ipynb` and `0.2_VGG_inference.ipynb` explain how to build VGG network from numpy files.
Then this code is sum up in `vgg.py`, to rapidly build networks.

It uses `tf.get_varaibel()` methods, to perform variable sharing.
See more details about this in the notebooks.


## MS COCO dataset

Some methods of style transfer, require a dataset of content images.
As many papers, I used the MS COCO dataset.
It contains about 80k images.

This dataset is available [here](http://mscoco.org/dataset/#download).

I used the [train set, from 2014 challenge](http://msvocds.blob.core.windows.net/coco2014/train2014.zip).

This file must placed in `data/COCO/` and extracted here.


In `coco.py` you will find a batch generator for COCO dataset.
See code for more details.


## WikiArt dataset

Some methods require a dataset of styled images.
As many papers I used the WikiArt dataset.


I used this previous work : http://github.com/eajena/wikiart-downloader

I made one change :
- there is no more `*.csv` files per style

```bash

cd data/WikiArt/

# to download a subset of wikiart ?
# see in download_featured.sh the list of styles
bash download_featured.sh


# to download much more images ...
bash download.sh
```

These scripts download into many sub-folders different paintings.
