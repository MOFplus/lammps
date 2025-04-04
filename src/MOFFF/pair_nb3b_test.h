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
PairStyle(nb3b/test,Pairnb3nTest);
// clang-format on
#else

#ifndef PAIR_NB3B_TEST
#define PAIR_NB3B_TEST

#include "pair.h"

namespace LAMMPS_NS {

class Pairnb3nTest : public Pair {
 public:
 Pairnb3nTest(class LAMMPS *);
  ~Pairnb3nTest() override;
  void compute(int, int) override;
  void settings(int, char **) override;
  void coeff(int, char **) override;
  void init_style() override;
  // double init_one(int, int) override;
  // double single(int, int, int, int, double, double, double, double &) override;

 protected:
  double cutoff;
  double k_a;
  double a_ref;

  int center_typeid;
  int edge_typeid;
  void allocate();
};

}    

#endif
#endif
