# Auto_Text_Generator

Retrained the model using  [Flickr](http://hockenmaier.cs.illinois.edu/DenotationGraph/) 30k image dataset which contains 158873 image caption.</br> 
Edited the `generate` function  to redirect the output to a file (optional). Pass `return_as_list=True` to the `generate` function to get the output as list (preferred).</br>

Requirements: [textgenrnn](https://github.com/minimaxir/textgenrnn) that uses Python3 and run on top on [Tensorflow](https://www.tensorflow.org/)

Attached:  Flickr 30k captions `In.txt` and the trained model `textgenrnn_weights.hdf5`</br></br>
Examples:</br>
Two teenagers are sitting on a sidewalk .</br>
A man is sitting on a bench looking at a park .</br>
Two people are standing on a rock .</br>
Man in a suit is in a field .</br>
Two young man in black shirt and a worker shorts playing a soccer ball .</br>
Two children are walking in front of a statue .</br></br>


**References:** </br>
[Original Github repo](https://github.com/minimaxir/textgenrnn)</br>
[Flickr](http://hockenmaier.cs.illinois.edu/DenotationGraph/)</br>
