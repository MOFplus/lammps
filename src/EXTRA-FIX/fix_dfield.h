/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Christina Payne (Vanderbilt U)
                        Stan Moore (Sandia) for dipole terms
                        Stephen J. Cox (University of Cambridge) for initial constant D implementation
                        See: https://github.com/uccasco/FiniteFields
                        Johannes P. Duerholt (RUB) for stress tensor and reformulation of terms
                        Babak Farhadi Jahromi (RUB) for nothing really
------------------------------------------------------------------------- */

#ifdef FIX_CLASS
// clang-format off
FixStyle(dfield,FixDfield);
// clang-format on
#else

#ifndef LMP_FIX_DFIELD_H
#define LMP_FIX_DFIELD_H

#include "fix.h"
#include "compute.h" // SJC: need this for the polarization

namespace LAMMPS_NS {

class FixDfield : public Fix {
  friend class FixQEqReaxFF;
  friend class FixQEqGauss; // CMC

 public:
  FixDfield(class LAMMPS *, int, char **);
  ~FixDfield() override;
  int setmask() override;
  void init() override;
  void setup(int) override;
  void min_setup(int) override;
  void post_force(int) override;
  void post_force_respa(int, int, int) override;
  void min_post_force(int) override;
  double memory_usage() override;
  double compute_scalar() override;
  double compute_vector(int) override;

 protected:
  double dx,dy,dz; // reduced components of the electric displacement field 
                   // as obtained by d_vector = volume * (cell^-1 <dot> D_vector)
  double Dx,Dy,Dz; // components of the electric displacement field
                   // mostly here for access from friend classes
  double OmegaPx, OmegaPy, OmegaPz; // SJC: Components of the
				    // intinerant polarization. These
				    // will likely be defined by a
				    // compute in the input file.
  int varflag;//,iregion;
  char *xstr_d,*ystr_d,*zstr_d,*estr;
  char *xstr_p,*ystr_p,*zstr_p;
  char *idregion;
  class Region *region;
  int xvar,yvar,zvar,evar,xstyle,ystyle,zstyle,estyle;
  int ilevel_respa;
  int qflag,muflag;
  int dxflag, dyflag, dzflag; // SJC: it might be useful to employ
			   // `hybrid' BCs, i.e. constant D in one
			   // direction, and tin-foil in the
			   // others. These flags indicate if a D
			   // field is being employed in a Cartesian
			   // direction or not.
  int maxatom;
  double **dfield;

  int force_flag;
  double fsum[4],fsum_all[4];

  // SJC: computes for the polarization
  class Compute *c_OmegaPx,*c_OmegaPy,*c_OmegaPz;
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Region ID for fix dfield does not exist

Self-explanatory.

E: Fix dfield requires atom attribute q or mu

The atom style defined does not have this attribute.

E: Variable name for fix dfield does not exist

Self-explanatory.

E: Variable for fix dfield is invalid style

The variable must be an equal- or atom-style variable.

E: Region ID for fix aveforce does not exist

Self-explanatory.

E: Fix dfield with dipoles cannot use atom-style variables

This option is not supported.

W: The minimizer does not re-orient dipoles when using fix dfield

This means that only the atom coordinates will be minimized,
not the orientation of the dipoles.

E: Cannot use variable energy with constant dfield in fix dfield

LAMMPS computes the energy itself when the E-field is constant.

E: Must use variable energy with fix dfield

You must define an energy when performing a minimization with a
variable E-field.

*/
