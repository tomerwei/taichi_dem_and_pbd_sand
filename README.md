# Improved Taichi DEM and PBD Sand Simulation
Includes: 1) A minimal DEM simulation demo written in Taichi, and 2) a minimal PBD sand simulation
We also include UI Control for: moving the right wall by left clicking with your mouse, and changing the angle of repose parameter which affects the tangential friction strength. 

Contrary to [another repo](https://github.com/taichi-dev/taichi_dem), we support tangential friction and include a slider for augmenting its affect on the particles.

<img src="https://raw.githubusercontent.com/taichi-dev/public_files/master/taichi_dem/demo.gif" height="270px">

## Installation
Make sure your `pip` is up-to-date:

```bash
$ pip3 install pip --upgrade
```

Assume you have a Python 3 environment, to install Taichi:

```bash
$ pip3 install -U taichi
```

To run the demo:

```bash
$ python dem.py
```

## Assumptions
The `sim.py` implements a minimal DEM solver with the following assumptions:

- All paricles are round circles with variable radius.
- normal and tangential forces between particles are included - angular or rolling forces are not included.
- The deformation of the particles is not included.

## Open missions
There are plenty of room for hacking! We suggest a few of them for you to start with:
- PBD sand does not replicate DEM angle of repose settings, either eith mui_static or mui_dynamic
- Support more DEM settings, shear forces, etc.
- Implement angular momentum of the particles, rolling friction
- ...

## References
["Unified particle physics for real-time applications"](https://mmacklin.com/uppfrta_preprint.pdf) Macklin, Miles, et al, ACM Transactions on Graphics (TOG) 33.4 (2014): 1-12.
["Particle-Based Simulation of Granular Materials"](http://wnbell.com/media/2005-07-SCA-Granular/BeYiMu2005.pdf) Bell, Nathan, et al, Proceedings of the 2005 ACM SIGGRAPH/Eurographics symposium on Computer animation. 2005.
