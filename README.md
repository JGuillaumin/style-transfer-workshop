

# Style Transfer by Deep Learning

This project includes source code (and more) presented during the workshop "Style Transfer by Deep Learning".

I spoke about style transfer in some MeetUps : 
 - [Artificial Intelligence Paris - MeetUp - May 18th, 2017](http://www.meetup.com/fr-FR/Artificial-Intelligence-Meetup-Paris/events/238029606/ "AI Paris MeetUp")
 - [Machine Learning Aix-Marseille - MeetUp - June 22th, 2017](http://www.meetup.com/fr-FR/Machine-learning-Aix-Marseille/events/240114846/ "ML Aix-Marseille MeetUp")

And I gave private sessions in companies about this topic, in particular : 
 - [Quantmetry](http://www.quantmetry.com/)
 - [Ercom](http://www.ercom.fr/)
 

## Install

In `install/` you will find :
 - The list of dependencies (mostly TensorFlow >= 1.x.y)
 - How to create a `conda` environment with all dependencies.
 - How to build a `docker` image with all dependencies (for CPU & GPU, with `nvidia-docker`).


## Downloads

This workshop uses pre-trained VGG networks (16 and 19 variants), and 2 datasets :
 - [MS COCO dataset](http://www.mscoco.org)
 - [WikiArt dataset](http://www.wikiart.org)
In `data/` you will find instructions for downloading these elements.


## Source Code and Notebooks

This project contains 4 main python files :

- `utils.py` : some functions for CNN visualization, TensorFlow tests, images manipulations and batch generator
- `vgg.py` : VGG16 and VGG19 implementations, with a scope factory for variable sharing


### Batch Generator

In `utils.py`, `BatchGenerator()` create a python generator of batches.

```python

COCOgenerator = BatchGenerator('data/COCO/train2014/',
                               reshape_mode='resize',
                               target_size=(256,256),
                               color_mode='rgb',
                               batch_size=32,
                               shuffle=True)

for it in range(ITERATIONS):
    batch = COCOgenerator.next()
    [...]

# or
for it, batch in enumerate(COCOgenerator):
    [...]

    it some_condition:
        break
```

**Performance** :
- custom batch generator with resize : `310 ms ± 2.8 ms per loop (mean ± std. dev. of 7 runs, 30 loops each)`
- custom batch generator with crop+resize : `373 ms ± 13.6 ms per loop (mean ± std. dev. of 7 runs, 30 loops each)`
- Keras ImageDataGenerator().flow_from_directory() : `1.4 s ± 23.5 ms per loop (mean ± std. dev. of 7 runs, 30 loops each)`

(I removed some Keras stuffs.)

#### Notebooks :

Then all methods and algorithms are presented within Jupyter notebooks.
It allows interactive programming (useful for education purpose).

- `0.0_test_batch_generator.ipynb` : some tests with batch generations. Comparison to Keras ImageDataGenerator.

- `0.1_VGG_construction.ipynb` : explains how to build VGG network with `tf.get_variable()`, for variable sharing.
- `0.2_VGG_inference.ipynb` : explains how to quickly build a VGG network and use it.

- `1.0_VGG_exploration.ipynb` : feature maps visualization, Google Dream approach for CNN understanding.
- `1.1_VGG_exploration_content.ipynb` : content reconstruction from feature maps, defines *content_loss*
- `1.2_VGG_exploration_style.ipynb` : style reconstruction from feature maps, defines *style_loss*

- `2.0_Gatys_method.ipynb` : implementation of method presented in [4]
- `2.0_Gatys_method_color.ipynb` : implementation of method presented in [10] for color preserving. (IN PROGRESS)
- `2.0_Gatys_method_video.ipynb` : implementation of method presented in [5] for video transformation. (IN PROGRESS)

- `3.0_feed_forward_method_IN.ipynb` : combination of several techniques from [6,7,8]. 

- `4.0_arbitrary_style_transfer.ipynb` : implementation of [11].
- `4.1_arbitrary_style_transfer_multi_gpu.ipynb` : implementation of [11] for multi-gpu systems (model parallelism). (IN PROGRESS)


I hope to find time to implement more techniques for style transfer using deep learning approaches.


## Slides

In `slides/` :

- `slides/StyleTransferWorkshop.pdf` : slides presented at "AI Paris MeetUp".
    - code screenshots are not from the current implementation.

- `slides/StyleTransferWorkshop_v3.pdf` : slides presented at "ML Aix-Marseille MeetUp"


## References


Some useful githubs :

- https://github.com/lengstrom/fast-style-transfer
- https://github.com/jcjohnson/fast-neural-style
- https://github.com/ghwatson/faststyle
- https://github.com/cysmith/neural-style-tf
- http://www.chioka.in/tensorflow-implementation-neural-algorithm-of-artistic-style
- https://github.com/albertlai/deep-style-transfer


Papers :

- [1] : K. Simonyan, A. Zisserman : “Very Deep Convolutional Networks for Large-Scale Image Recognition”, 2014, arXiv:1409.1556
- [2] : About Deep Dream visualization technique :  “Inceptionism: Going Deeper into Neural Networks”
- [3] : M. Zeiler, R. Fergus: Visualizing and Understanding Convolutional Networks, 2013 arXiv:1311.2901 
- [4] : L. Gatys, A. Ecker, M. Bethge : A neural algorithm of artistic style, 2015, arXiv:1508.06576
- [5] : M. Ruder, A. Dosovitskiy, T. Brox : Artistic style transfer for video, 2016, arXiv:1604.08610 
- [6] : D. Ulyanov et al : Texture Networks: Feed-forward Synthesis of Textures and Stylized Images, 2016, arXiv:1603.03417
- [7] : D. Ulyanov et al : Instance Normalization: The Missing Ingredient for Fast Stylization, 2016, arXiv:1607.08022
- [8] : J. Johnson et al : Perceptual losses for real-time style transfer and super-resolution, 2016, arXiv:1603.08155
- [9] : V. Dumoulin et al : A learned representation for artistic style, 2017, arXiv:1610.07629
- [10] : Gatys et al : Preserving color in Neural Artistic Style Transfer, 2016, arXiv:1606.05897
- [11]  X. Huang and S. Belongie : Arbitrary Style Transfer in real-time with AdaIN, 2017, arXiv:1703.06868