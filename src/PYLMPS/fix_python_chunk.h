/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.

    RS RUB 2023
    part of pylmps

    Target: a fix that computes COMs from chunks and provides them as a 
    communicated array in python. In python we get forces fcom on these 
    COMs with some potential and provide forces to atoms

    derived from fix_spring_chunk and fix_python_invoke

------------------------------------------------------------------------- */

#ifdef FIX_CLASS
// clang-format off
FixStyle(python/chunk,FixPythonChunk);
// clang-format on
#else

#ifndef LMP_FIX_PYTHON_CHUNK_H
#define LMP_FIX_PYTHON_CHUNK_H

#include "fix.h"
#include <Python.h>

namespace LAMMPS_NS {

class FixPythonChunk : public Fix {
 public:
  FixPythonChunk(class LAMMPS *, int, char **);
  ~FixPythonChunk() override;
  int setmask() override;
  void init() override;
  void setup(int) override;
  void min_setup(int) override;
  void post_force(int) override;
  void post_force_respa(int, int, int) override; // check RESPA later
  void min_post_force(int) override;
  // void write_restart(FILE *) override; // we handle our restart ourselves
  // void restart(char *) override;
  double compute_scalar() override;

 private:
  // int ilevel_respa;

  void *lmpPtr;
  void *pFunc;
  
  double epychunk;
  char *idchunk, *idcom;

  PyObject * py_com;
  PyObject * py_fcom;

  int nchunk;
  double **fcom;

  class ComputeChunkAtom *cchunk;
  class ComputeCOMChunk *ccom;
};

}    // namespace LAMMPS_NS

#endif
#endif
