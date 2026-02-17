/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.org, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
   Contributing author: Pierre de Buyl (KU Leuven)
                        Rochus Schmid (RUB)
                        Note: this is a rip off of Pierre de Buyl's h5md dump
                              to write hdf5 based mfp5 files. Thanks to Pierre for the clear code!
------------------------------------------------------------------------- */

#ifdef DUMP_CLASS
// clang-format off
DumpStyle(mfp5,DumpMFP5)
// clang-format on
#else

#ifndef LMP_DUMP_MFP5_H
#define LMP_DUMP_MFP5_H

#include "dump.h"
#include "hdf5.h"

#include "fix.h"
#include "fix_reaxff_bonds.h"


namespace LAMMPS_NS {

class DumpMFP5 : public Dump {
 public:
  DumpMFP5(class LAMMPS *, int, char**);
  ~DumpMFP5() override;

 private:
  char *stage_name;
  int natoms,ntotal;
  int unwrap_flag;            // 1 if atom coords are unwrapped, 0 if no
  hid_t mfp5file;
  long long int hdf5id;      // use this to store the hdf5 file id that is passed in via the dump command from pylmps
  hid_t stage_group, traj_group, restart_group;
  hid_t rest_xyz_dset, rest_vel_dset, rest_cell_dset, rest_img_dset;

  hid_t xyz_dset;
  hid_t img_dset;
  hid_t vel_dset;
  hid_t forces_dset;
  hid_t charges_dset;
  hid_t cell_dset;
  // hid_t thermo_dset;
  hid_t bondtab_dset;
  hid_t bondord_dset;

  // data arrays and intervals
  int every_dump;
  double *dump_xyz;
  int every_xyz;
  
  int *dump_img;
  int every_image;
  double *dump_vel;
  int every_vel;
  double *dump_forces;
  int every_forces;
  double *dump_charges;
  int every_charges;
  int every_cell;

  int every_restart;
  // int every_thermo;
  int every_bond;

  int dump_count;

  FixReaxFFBonds *rxbfix;

  void init_style() override;
  int modify_param(int, char **) override;
  void openfile() override;
  void write_header(bigint) override{};
  void pack(tagint *) override;
  void write_data(int, double *) override;

  void write_frame();
  int append_data(hid_t, int, double *);
  int write_data(hid_t, int, double *);
  int append_data_int(hid_t, int, int *);
  int write_data_int(hid_t, int, int *);
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Invalid number of arguments in dump h5md

Make sure that each data item (position, etc.) is followed by a dump
interval.

E: Dump h5md requires sorting by atom ID

Use the dump_modify sort command to enable this.

E: Cannot use variable every setting for dump xtc

The format of this file requires snapshots at regular intervals.

E: Cannot change dump_modify every for dump xtc

The frequency of writing dump xtc snapshots cannot be changed.

E: Cannot set file_from in dump h5md after box or create_group

The file_from option modifies the box and create_group options and
they must appear after file_from if used.

*/
