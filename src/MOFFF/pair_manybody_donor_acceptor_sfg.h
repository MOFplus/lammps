/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS
// clang-format off
PairStyle(manybody/donor/acceptor/sfg,PairManybodyDonorAcceptorSFG);
// clang-format on
#else

#ifndef PAIR_MANYBODY_DONOR_ACCEPTOR_SFG
#define PAIR_MANYBODY_DONOR_ACCEPTOR_SFG

#include "pair.h"

namespace LAMMPS_NS {

class PairManybodyDonorAcceptorSFG : public Pair {
 public:
  PairManybodyDonorAcceptorSFG(class LAMMPS *);
  ~PairManybodyDonorAcceptorSFG() override;
  void compute(int, int) override;
  void settings(int, char **) override;
  void coeff(int, char **) override;
  void init_style() override;
  // double init_one(int, int) override;
  // double single(int, int, int, int, double, double, double, double &) override;

 protected:
  double cutoff_dsf;


  struct Param {
    double dp_0, ds_0, a_p, a_s, rp_0, rs_0;
    double cutoff_dsf;
  };

  Param *params;    // parameter set for an I-J
  int nparams;      // number of parameters read
  int maxparam;

  int donor_typeid;
  int acceptor_typeid;
  int **type2param;

  void allocate();
};

}    

#endif
#endif
