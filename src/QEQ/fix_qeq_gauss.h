// clang-format off
/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Babak Farhadi Jahromi, CMC group,
   Ruhr-Universitaet Bochum 

   Based on fix qeq/reaxff by Hasan Metin Aktulga and fix qeq by Ray Shan

   Therefore, please cite the related publication:
   H. M. Aktulga, J. C. Fogarty, S. A. Pandit, A. Y. Grama,
   "Parallel Reactive Molecular Dynamics: Numerical Methods and
   Algorithmic Techniques", Parallel Computing, in press.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS
// clang-format off
FixStyle(qeq/gauss,FixQEqGauss);
// clang-format on
#else

#ifndef LMP_FIX_QEQ_GAUSS_H
#define LMP_FIX_QEQ_GAUSS_H

#include "fix.h"

#define SQR(x) ((x) * (x))
#define DANGER_ZONE 0.90
#define MIN_CAP 50
#define SAFE_ZONE 1.2
#define MIN_NBRS 100

namespace LAMMPS_NS {

class FixQEqGauss : public Fix {
 public:
  FixQEqGauss(class LAMMPS *, int, char **);
  ~FixQEqGauss() override;
  int setmask() override;
  void post_constructor() override;
  void init() override;
  void init_list(int, class NeighList *) override;
  virtual void init_storage();
  void setup_pre_force(int) override;
  void pre_force(int) override;

  void setup_pre_force_respa(int, int) override;
  void pre_force_respa(int, int, int) override;

  void min_setup_pre_force(int);
  void min_pre_force(int) override;

  double compute_scalar() override;

  double get_cutoff() { return cutoff; }
  double *get_chi() { return chi; }
  double *get_eta() { return eta; }
  double *get_zeta() { return zeta; }

 protected:
  int nevery;
  int matvecs;
  int nn, m_fill;
  int n_cap, nmax, m_cap;
  int pack_flag;
  int nlevels_respa;
  class NeighList *list;
  class FixEfield *efield;
  int *ilist, *jlist, *numneigh, **firstneigh;

  double tolerance;    // tolerance for the norm of the rel residual in CG
  double cutoff, cutoff_sq;    // neighbor cutoff

  double *chi, *eta, *zeta, *zcore;    // qeq parameters
  double *chizj;

  // fictitious charges

  double *s, *t;
  double **s_hist, **t_hist;
  int nprev;

  typedef struct {
    int n, m;
    int *firstnbr;
    int *numnbrs;
    int *jlist;
    double *val;
  } sparse_matrix;

  sparse_matrix H;
  double *Hdia_inv;
  double *b_s, *b_t;
  double *b_prc, *b_prm;
  double *chi_field;

  //CG storage
  double *p, *q, *r, *d;
  int imax, maxwarn;

  // constant Dfield
  class FixDfield *dfield;
  double *chi_dfield;
  class FixStoreState *store;
  class ComputeDisplaceAtom *displace;
  double **x0;
  // this is dense, this is horrible
  double **H_dfield;
  int **H_dfield_jarray;
  // actually it turns somewhat sparse when parallel, keep this for later
  //sparse_matrix H;
  virtual void compute_H_dfield();
  double calculate_H_dfield(int, int);
  virtual void get_chi_dfield();

  char *pertype_option;    // argument to determine how per-type info is obtained
  virtual void pertype_parameters(char *);
  virtual void allocate_storage();
  virtual void deallocate_storage();
  void reallocate_storage();
  virtual void allocate_matrix();
  virtual void deallocate_matrix();
  void reallocate_matrix();

  virtual void init_matvec();
  void init_H();
  virtual void compute_H();
  double calculate_H(double, double, double);
  double calculate_H_wolf(double, double, double);
  double calculate_H_dsf(double, double, double);
  virtual void calculate_Q();

  virtual int CG(double *, double *);
  virtual void sparse_matvec(sparse_matrix *, double *, double *);

  int pack_forward_comm(int, int *, double *, int, int *) override;
  void unpack_forward_comm(int, int, double *) override;
  int pack_reverse_comm(int, int, double *) override;
  void unpack_reverse_comm(int, int *, double *) override;
  double memory_usage() override;
  void grow_arrays(int) override;
  void copy_arrays(int, int, int) override;
  int pack_exchange(int, double *) override;
  int unpack_exchange(int, double *) override;

  virtual double parallel_norm(double *, int);
  virtual double parallel_dot(double *, double *, int);
  virtual double parallel_vector_acc(double *, int);

  virtual void vector_sum(double *, double, double *, double, double *, int);
  virtual void vector_add(double *, double, double *, int);

  virtual void get_chi_field();

  // dual CG support
  int dual_enabled;            // 0: Original, separate s & t optimization; 1: dual optimization
  int matvecs_s, matvecs_t;    // Iteration count for each system
};

}    // namespace LAMMPS_NS

#endif
#endif
