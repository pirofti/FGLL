# Factor Graph Optimization for Leak Localization in Water Distribution Networks

![Factor Graph for Leak Localization](figures/fgo-design.png?raw=true)

Implementation and [experimental data](network_data) for the [paper](https://arxiv.org/pdf/2509.10982)

> P. Irofti, L. Romero-Ben, F. Stoican, and V. Puig,
“Factor Graph Optimization for Leak Localization in Water
Distribution Networks,"
pp. 1--12, 2025.

If you use our work in your research, please cite as:
```
@article{IRSP25_fgll,
  author = {Irofti, P. and Romero-Ben, L. and Stoican, F. and Puig, V.},
  title = {Factor Graph Optimization for Leak Localization in Water
Distribution Networks},
  year = {2025},
  pages = {1-12},
  eprint = {2509.10982},
  archiveprefix = {arXiv},
}
```

## Prerequisite
Before running make sure you have installed the Python packages:
* [numpy](https://numpy.org/)
* [scipy](https://scipy.org/)
* [gtsam](https://gtsam.org/)
* [wntr](https://github.com/USEPA/WNTR)

## Usage
Run [test_FGLL.py](test_FGLL.py) and set the network parameter to `Modena`, `LTOWN` or `toy_example`. Default is `Modena`.

## Description
Detecting and localizing leaks in water distribution network systems is an important topic with direct environmental, economic, and social impact.
Our paper is the first to explore the use of factor graph optimization techniques for leak localization in water distribution networks,
enabling us to perform sensor fusion between pressure and demand sensor readings
and to estimate the network's temporal and structural state evolution across all network nodes.
The methodology introduces specific water network factors and proposes a new architecture composed of two factor graphs:
a leak-free state estimation factor graph and a leak localization factor graph.
When a new sensor reading is obtained,
unlike Kalman and other interpolation-based methods,
which estimate only the current network state,
factor graphs update both current and past states.
Results on Modena, L-TOWN and synthetic networks show that factor graphs are much faster than nonlinear Kalman-based alternatives such as the UKF,
while also providing improvements in localization compared to state-of-the-art estimation-localization approaches.

## Contents
1. The **Factor Graph Leak Localization** (FGLL) algorithm is in [FGLL.py](FGLL.py).

2. The custom **water factors** are in [water_factors.py](water_factors.py).

3. Specific water distribution network data are in [network_data](network_data).

## Results

In the paper we compared our results with [GHR-S](https://www.sciencedirect.com/science/article/abs/pii/S0043135423001823?via%3Dihub), [GSI](https://github.com/luisromeroben/PhD/tree/master/Chapter3) and [UKF-AW-GSI](https://github.com/luisromeroben/D-UKF-AW-GSI).

![Normalized leak metric for each potential leak](figures/matrix_tile_texample.png?raw=true)

Description: Normalized leak metric for each potential leak, comparing GHR-S, GSI, UKF-AW-GSI and FGLL. Each image encodes a colour code of the normalized metric of a node (x-axis) in a leak scenario (y-axis).
