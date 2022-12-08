# beyonce-shallot
This repository contains all the necessary information to produce the figures and perform the analysis that is described in <strong>van Dam & Kenworthy 2023</strong>. 

&nbsp; 

## Introduction

<em>BeyonCE</em> (Beyond Common Eclipsers) is a python package that is used to simplify the exploration of the parameter space that describes circumplanetary ring system transits.
It is split up into two main modules:

1. <em>Shallot Explorer</em>
2. <em>Ring Divide Fitter</em>

This repository concerns itself with the Shallot Explorer.

<em>Shallot Explorer</em> is the module contained and described here.
It is used to reduce the very large parameter space spanned by circumplanetary ring system transits.

| Sub-System       | Parameter                   |
|------------------|-----------------------------|
| Star             | Radius                      |
|                  | Limb-Darkening              |
| Planet           | Radius                      |
| Rings            | Inner Radius                |
|                  | Outer Radius                |
|                  | Opacity                     |
| System Geometry  | Inclination                 |
|                  | Tilt                        |
| Transit Geometry | Impact Parameter            |
|                  | Time of Maximum Occultation |
|                  | Eclipse Duration            | 

It does this by producing a 3D grid with coordinate axes: midpoint of the disc, or planet location $(\delta x, \delta y)$ , and the radius scale factor $(f_R)$.
The radius scale factor is a value (> 1) that relates the radius of the disc to the smallest possible radius of a disc that is centred at $(\delta x, \delta y)$ and has a given eclipse duration.

It can subsequently cut down the parameter space by introducing two astrophysical restrictions, namely:
1. Hill Sphere Stability
2. Valid Gradients

The Hill sphere stability refers to the fact that the disc around a certain mass anchor body must be contained by that body's Hill sphere to be a stable phenomenon.

Valid Gradients refers to the fact that gradients measured in the light curve of the transit are related to the physical geometry of the transiting disc, and any discs that do not permit a given light curve gradient are removed.

&nbsp; 

## Contents

This repository contains
- <em>Shallot Explorer</em> code (with unit tests)
- Scripts for producing figures in the relevant paper
- Jupyter notebooks and data for case study analysis
- Figures in the paper
- The paper itself
