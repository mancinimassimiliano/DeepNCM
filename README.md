# DeepNCM
PyTorch implementation of "DeepNCM: Deep Nearest Class Mean Classifiers" by Samantha Guerriero, Barbara Caputo and Thomas Mensink, ICLR Workshop 2018 (https://openreview.net/pdf?id=rkPLZ4JPM)

To run the code with standard DeepNCM on Cifar100, just run:
```
python training_ncm.py --dataset=100
```
change to ```dataset=10``` for test on Cifar10. For visualization purposes, visdom is employed. To disable it just add the option ```--no_vis``` when to the above command.

The implementation of the layer is on *ncm_layer.py* . 
To apply the layer to any network, just replace the standard linear classifier with the NCM one and be aware of this 2 differences:
* The forward step has 2 phases, a first phase which computes the features relative to the given images and a second phase which computes the class scores (-distances).
* After the loss computation and backward, the means of the classifier must be updated.

For the incremental case, other 2 differences are present:
* The forward step must be anticipated by a preparation step which adds to the network the space for novel class means.
* The targets must be converted to the order in which the classifier have seen the classes, to align predictions and labels.

For both cases examples can be found in the relative training files and networks.


