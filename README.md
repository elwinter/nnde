# nnde

`nnde` is a package of Python modules which implement a set of neural networks which solve ordinary and partial differential equations using technique of [Lagaris et al. (1998)](https://dx.doi.org/10.1109/72.712178).

The `nnde` package provides a pure-Python implementation of one of the earliest approaches to using neural networks to solve differential equations - the trial function method [Lagaris et al. (1998)](https://dx.doi.org/10.1109/72.712178). It was initially developed primarily as a vehicle for understanding the internal workings of feedforward neural networks, without the restrictions imposed by an existing neural network framework. The target audiences are machine learning researchers who want to develop an understanding of the basic steps needed to solve differential equations with neural networks and researchers from general engineering fields who want to solve a differential equation without using a complex neural network library.

The package is described in [Winter and Weigel, 2021](https://github.com/elwinter/
/blob/master/paper.pdf). Additional details on the method are given in [Winter 2020](https://github.com/elwinter/proposal/blob/master/proposal.pdf).

Demos are available in the repository [`nnde_demos`](https://github.com/elwinter/nnde_demos).

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
python setup.py test  # Deprecated
pytest  # Preferred
```

# An overview of nnde

The `nnde` package was developed as a tool to help understand how neural networks can be used to solve differential equations. The effort was originally inspired by work done in the 1990s showing how feedforward neural networks could be used as universal approximators. That work led to efforts to show how neural networks could be applied to differential equations. A good example of this technique can be found in Lagaris et al (1993).

The basic technique is straightforward:

* Arrange the differential equation in a standard form:

  G(x, y, dy/dx, ...) = 0

* Define a trial solution `y_trial` that can be substituted into the differential equation. This trial solution includes a component computed by a neural network, as well as a term which incorporates all boundary conditions.

* Use the analytical definition of `y_trial` to determine the analytical forms of the various derivatives of `y` used in the differential equation.

* Define the structure of the neural network so that it has one input per independent variable, and a single output.

* Using a set of training points defined on the domain of the differential equation, run the network and use the output to compute the value of the trial solution and its derivatives.

* Compute the value of the standardized differential equation `G()` at each training point.

* Compute the loss function as the sum of the squared values of `G()`.

* Train a neural network to solve the equation for the trial solution by minimizing the loss function at each training point: `L = SUM(G_i^2)`. This is done by adjusting the network parameters (weights and biases) until a satifactory solution is obtained.

The `nnde` package is divided into 3 major packages (`differentialequation`, `neuralnetwork`, and `trialfunction`), and two auxiliary packages (`exceptions` and `math`).

The `differentialequation` package is divided into two sub-packages: `ode` (for ordinary differential equations) and `pde` (for partial differential equations). These sub-packages are very lightweight - they are composed of abstract classes used to define the methods that the user-defined differential equation  must implement. These are primarily methods to evaluate the differential equation itself, and the various derivatives required for its evaluation. Each of these functions depends on the training points, and computes the values and derivatives of the trial solution during evaluation The ode and pde packages provide classes for 1st-order ODE initial-value problems, 1st-order PDE initial value problems, and diffusion problems in 1, 2, and 3 spatial dimensions. These classes can be used as-is to solve the corresponding equation types by defining the required methods.

The `neuralnetwork` package provides the core of the `nnde` functionality. Each module provides a customizable neural network tailored to the needs of a specific problem type. Currently support types are the same as those supported in the `differentialequation` package. A different class was used for each equation type because the details of the computation of the network output differ slightly for each equation type.

The `trialfunction` package provides a previously unavailable capability: determine the structure of the boundary condition component of the trial solution automatically. The algorithm for determining the form of this component of the trial solution is recursive, and the number of terms grows rapidly as the dimensionality of the problem is increased. However, if the user provides the boundary conditions (as another set of functions), these modules can automatically construct the boundary condition term, greatly easing the problem definition burden on the user. The modules also provide the option to short-circuit this process by allowing the user to define an optimized form of the boundary condition function that can greatly reduce the required amount of computation. Trial function classes are provided for diffusion problems in 1, 2, and 3 spatial dimensions. The code which solves 1st-order ODE IVP uses a trivial form of the boundary condition component (a constant), and therefore its use is coded directly, without a separate class.

A Jupyter notebook ([`tutorial.ipynb`](https://github.com/elwinter/nnde/tree/master/examples)) providing a structured walkthrough of the process of defining and solving a problem using the `nnde` package is available in the `examples` directory.

# Contribute

If you discover bugs in the `nnde` package, please create an issue at the project repository on GitHub at https://github.com/elwinter/nnde.

If you find the nnde package useful, we welcome your contributions of code and documentation. To contribute, fork the repository on GitHub, and submit a pull request at https://github.com/elwinter/nnde.

# Contact

Submit bug reports and feature requests on the [repository issue tracker](https://github.com/elwinter/nnde/issues).

Eric Winter <eric.winter62@gmail.com>
