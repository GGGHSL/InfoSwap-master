## InfoSwap: Information Bottleneck Disentanglement for Identity Swapping (Official PyTorch Implementation)

### License

Copyright (C) 2021, CRIPAC, NLPR, CASIA. All rights reserved. Licensed under the CC BY-NC-SA 4.0 license ([https://creativecommons.org/licenses/by-nc-sa/4.0/](https://creativecommons.org/licenses/by-nc-sa/4.0/)).

### Getting started

#### Requirements
See [requirements.txt](./requirements.txt), tested on Linux platforms. 

For pre-trained models, please [send a request email to us](mailto:grace.heseri@gmail.com) and describe in detail your purpose of using the models.

#### Example Usage

Clone this repo: 

```shell script
git clone https://github.com/GGGHSL/InfoSwap-master.git
cd InfoSwap-master
```

Run the following command: 
```shell script    
python inference_demo.py -src [YOUR SOURCE IMAGE] -tar [YOUR DIR OF TARGET IMAGES] -save [YOUR SAVE DIR] --ib_mode [CHOICES: smooth, no_smooth]
```
The results are stored in `results_[INFERENCE_DATE]` folder.
