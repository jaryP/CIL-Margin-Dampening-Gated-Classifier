# [Class incremental learning with probability dampening and cascaded gated classifier](https://arxiv.org/abs/2402.01262)
[Jary Pomponi](https://jarypomponi.com/), [Alessio Devoto](https://alessiodevoto.github.io/),  [Simone Scardapane](https://www.sscardapane.it/)

### Abstract
Humans are capable of acquiring new knowledge and transferring learned knowledge into different domains, incurring a small forgetting. The same ability, called Continual Learning, is challenging to achieve when operating with neural networks due to the forgetting affecting past learned tasks when learning new ones. This forgetting can be mitigated by replaying stored samples from past tasks, but a large memory size may be needed for long sequences of tasks; moreover, this could lead to overfitting on saved samples. In this paper, we propose a novel regularisation approach and a novel incremental classifier called, respectively, \nreg and \nhead. The first combines a soft constraint and a knowledge distillation approach to preserve past learned knowledge while allowing the model to learn new patterns effectively. The latter is a gated incremental classifier, helping the model modify past learned classes without directly interfering with them. This is achieved by modifying the output of the model with auxiliary scaling functions. We empirically show that our approach performs well on many benchmarks against multiple well-established baselines, and we also study each component of our proposal and how the combinations of such components affect the final results.

### Main Dependencies
* pytorch==2.0.1
* python=3.8.18
* torchvision==0.15.2
* hydra-core==1.3.2
* avalanche-lib==0.4
* wandb==0.16.0

### Proposal and experiments

In the paper we proposed a regularisation schema, Margin Dampening, coupled with a classifier head, called Cascaded Scaling Classifier. 
You can find both in the folder proposal. 

To run the experiments, you can use the files contained in the folder bash, which are grouped by the scenario.

Use such files also to check the correct syntax to run custom experiments.

[//]: # (The folder './config/' contains all the yaml files used for the experiments presented in the paper. )

[//]: # ()
[//]: # (The folders './config/optimizers' and './config/training' contain, respectively, the files which contain the optimizers and the training strategies. )

[//]: # ()
[//]: # (The folder './config/experiments/classification' contains all the files used for the ensemble experiments, while './config/experiments/classification' contains the ones used in the CL scenarios.)

[//]: # ()
[//]: # (### Training)

[//]: # (We have teo training files:)

[//]: # ()
[//]: # (* main.py: to be used only with config files from './config/experiments/classification')

[//]: # (* main_cl.py: to be used only with config files from './config/experiments/cl')

[//]: # ()
[//]: # (Bot scripts accept any number of training files, which are processed sequentially, and also an optional flag --device [integer|cpu] that can be used to specify the device &#40;otherwise the one present in each config file is used&#41;.)

[//]: # ()
[//]: # (Please refer to the yaml files to understand how they can be formatted, and to the methods to understand the parameters that can be used.)

[//]: # ()
[//]: # (If you want to use TinyImagenet you need to download and preprocess it first, using the script 'tinyimagenet_download.sh'.)

### Cite

Please cite our work if you find it useful:

```
@misc{pomponi2024cascaded,
    title={Cascaded Scaling Classifier: class incremental learning with probability scaling},
    author={Jary Pomponi and Alessio Devoto and Simone Scardapane},
    year={2024},
    eprint={2402.01262},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```
