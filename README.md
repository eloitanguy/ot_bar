# Optimal Transport Barycenters

This repository provides code for the fixed-point approach to OT barycenters
with arbitrary cost functions.

The main function to use in practice is
`ot_bar.solvers.solve_OT_barycenter_fixed_point` which solves the OT barycentre
problem using the (barycentric) fixed-point method.

To install required packages:

    pip install -r requirements.txt

To install this repository as an editable package:

    pip install -e .