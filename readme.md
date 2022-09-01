## Uncertainty as a Proxy of the Generalization Error for Marine Species Identification
Repo still in progress...

Thesis Dissertation at: [[here](imgs/TDM_David.pdf)]

![example](imgs/echinaster.jpg)


### Introduction
Conservation of marine species is a must to control the levels of contamination and degradation of underwater ecosystems,
which get worse every year. One of the best ways of monitoring aquatic ecosystems is by using a Remotely Operated Underwater
Vehicle to collect images. However, the main drawback is the large amount of data that has to be annotated by specialists. In this
thesis, we propose to go one step further and use a deep learning system which reports, in addition to the deterministic decision,
uncertainty estimations to identify misclassifications. We test several well-known and a novel uncertainty metric which evaluates
the quality of the estimations concerning their ordering. Furthermore, we also propose a novel protocol which helps reduce the
workload of annotating images using the uncertainty as a proxy of the generalization error.

In this repository, I propose two model wrappers to both compute uncertainty with MC dropout methods and Class Activation Maps with a wide variety of methods.
Furthermore, the uncertainty metrics and plots can be found.