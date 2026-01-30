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

#ifdef FIX_CLASS

FixStyle(press/crescale,FixPressCRescale)

#else

#ifndef LMP_FIX_PRESS_CRESCALE_H
#define LMP_FIX_PRESS_CRESCALE_H

#include "fix.h"

namespace LAMMPS_NS {

class FixPressCRescale : public Fix {
 public:
  FixPressCRescale(class LAMMPS *, int, char **);
  ~FixPressCRescale();
  int setmask();
  void init();
  void setup(int);
  void end_of_step();
  int modify_param(int, char **);

 protected:
  int dimension,which;
  double bulkmodulus;
  double ktv;
  double noise_prefactor, determ_prefactor;
  double vol0;                          // reference volume

  double t_start,t_stop;
  double t_target;

  int pstyle,pcouple,allremap;
  int p_flag[6];                        // 1 if control P on this dim, 0 if not
  double p_start[6],p_stop[6];
  double p_period[6],p_target[6];
  double p_period_global;
  double p_current[6];
  double p_current_full[3][3];          // full symmetric pressure tensor 
  double dilation[6],dilation_inv[6];   // rescaling matrices 
  double randoms[3][3];                 // random numbers tensor
  int kspace_flag;                      // 1 if KSpace invoked, 0 if not
  int nrigid;                           // number of rigid fixes
  int *rfix;                            // indices of rigid fixes

  char *id_temp,*id_press;
  class Compute *temperature,*pressure;
  int tflag,pflag;

  double p_hydro;                  // hydrostatic target pressure
  int pdim;                        // number of barostatted dims

  double sigma[6];                 // scaled target stress
  double fdev[6];                  // deviatoric force on barostat   
  double fdev_full[3][3];          // full symmetric deviatoric force
  int deviatoric_flag;             // 0 if target stress tensor is hydrostatic
  double h0_inv[6];                // h_inv of reference (zero strain) box
  int nreset_h0;                   // interval for resetting h0

  class RanMars *random;
  double fixedpoint[3];            // location of dilation fixed-point

  void couple();
  void remap();

  void compute_temp_target();
  void compute_press_target();
  void compute_sigma();
  void compute_deviatoric();

  void voigt2fullmatrix(double[6], double[3][3], bool);
  void matrix_prod(double[3][3], double[3][3], double[3][3]);
  void matrix_prod(double[6], double[6], double[6]);
  void backrotate(double[3][3], double[6]);
  void vector_matrix_prod(double[3], double[6], double[3]);
  void inverse_matrix(double[6], double[6]);
};

}


#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Invalid fix press/crescale for a 2d simulation

The z component of pressure cannot be controlled for a 2d mofdel.

E: Invalid fix press/crescale pressure settings

Settings for coupled dimensions must be the same.

E: Cannot use fix press/crescale on a non-periodic dimension

Self-explanatory.

E: Fix press/crescaledamping parameters must be > 0.0

Self-explanatory.

E: Cannot use fix press/crescale with triclinic box

Self-explanatory.

E: Cannot use fix press/crescale and fix deform on same component of stress tensor

These commands both change the box size/shape, so you cannot use both
together.

E: Temperature ID for fix press/crescale does not exist

Self-explanatory.

E: Pressure ID for fix press/crescale does not exist

The compute ID needed to compute pressure for the fix does not
exist.

E: Could not find fix_modify temperature ID

The compute ID for computing temperature does not exist.

E: Fix_modify temperature ID does not compute temperature

The compute ID assigned to the fix must compute temperature.

W: Temperature for NPT is not for group all

User-assigned temperature to NPT fix does not compute temperature for
all atoms.  Since NPT computes a global pressure, the kinetic energy
contribution from the temperature is assumed to also be for all atoms.
Thus the pressure used by NPT could be inaccurate.

E: Could not find fix_modify pressure ID

The compute ID for computing pressure does not exist.

E: Fix_modify pressure ID does not compute pressure

The compute ID assigned to the fix must compute pressure.

E: Cannot use fix press/crescale without temp specification

'temp' keyword must be used in the form 
    fix press/berendsen ... temp tstart tstop
where tstart and tstop are the starting and final temperatures of 
the thermostat coupled with CRescale

E: Cannot use fix press/crescale with different pressure damping parameters if box is triclinic

Self-explanatory.

*/
