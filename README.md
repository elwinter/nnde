# nnde

nnde is a package of Python modules which implement a set of neural networks which solve ordinary and partial differential equations using technique of [Lagaris et al. (1998)](https://dx.doi.org/10.1109/72.712178).

The nnde package provides a pure-Python implementation of one of the earliest approaches to using neural networks to solve differential equations - the trial function method [Lagaris et al. (1998)](https://dx.doi.org/10.1109/72.712178). It was initially developed primarily as a vehicle for understanding the internal workings of feedforward neural networks, without the restrictions imposed by an existing neural network framework. The target audiences are machine learning researchers who want to develop an understanding of the basic steps needed to solve differential equations with neural networks and researchers from general engineering fields who want to solve a differential equation without using a complex neural network library.

The package is described in [Winter and Weigel, 2021](https://github.com/elwinter/nnde/blob/master/paper.pdf). Additional details on the method are given in [Winter 2020](https://github.com/elwinter/proposal/blob/master/proposal.pdf).

Demos are available in the repository [nnde_demos](https://github.com/elwinter/nnde_demos).

# Install and Use

```bash
pip install nnde --update
git clone https://github.com/elwinter/nnde_demos
pip install matplotlib # optional
cd nnde_demos; python lagaris01_demo.py
```

# Developer

```bash
git clone https://github.com/elwinter/nnde
cd nnde
pip install -e .
python setup.py test
```

# Contact

Submit bug reports and feature requests on the [repository issue tracker](https://github.com/elwinter/nnde/issues).

Eric Winter <eric.winter62@gmail.com>
