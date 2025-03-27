# Optimal Transport Barycenters

This repository provides code for the fixed-point approach to OT barycenters
with arbitrary cost functions, and our implementation for [our preprint](https://arxiv.org/abs/2407.13445).

The main function to use in practice is
`ot_bar.solvers.solve_OT_barycenter_fixed_point` which solves the OT barycentre
problem using the (barycentric) fixed-point method.

To install required packages:

    pip install -r requirements.txt

To install this repository as an editable package:

    pip install -e .

### To cite this work:

    @misc{tanguy2025constrainedapproximateoptimaltransport,
        title={Constrained Approximate Optimal Transport Maps}, 
        author={Eloi Tanguy and Agnès Desolneux and Julie Delon},
        year={2025},
        eprint={2407.13445},
        archivePrefix={arXiv},
        primaryClass={math.OC},
        url={https://arxiv.org/abs/2407.13445}, 
    }