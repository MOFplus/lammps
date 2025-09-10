// clang-format off
/* ----------------------------------------------------------------------
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
   Contributing author: Babak Farhadi Jahromi, CMC Group,
   Ruhr-Universitaet Bochum

   Based on fix acks2/reaxff by Stan Moore
------------------------------------------------------------------------- */

#include "fix_acks2_gauss.h"

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "neigh_list.h"
#include "pair.h"
#include "text_file_reader.h"
#include "update.h"

#include <cmath>
#include <cstring>

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixACKS2Gauss::FixACKS2Gauss(LAMMPS *lmp, int narg, char **arg) :
  FixQEqGauss(lmp, narg, arg)
{
  X_diag = nullptr;
  Xdia_inv = nullptr;

  // BiCGStab
  g = nullptr;
  q_hat = nullptr;
  r_hat = nullptr;
  y = nullptr;
  z = nullptr;

  // X matrix
  X.firstnbr = nullptr;
  X.numnbrs = nullptr;
  X.jlist = nullptr;
  X.val = nullptr;

  // KS potential vector
  u = nullptr;

  special_local = nullptr;
  nspecial_local = nullptr;

  // Update comm sizes for this fix
  comm_forward = comm_reverse = 2;

  s_hist_X = s_hist_last = nullptr;

  last_rows_rank = 0;
  last_rows_flag = (comm->me == last_rows_rank);
}

/* ---------------------------------------------------------------------- */

FixACKS2Gauss::~FixACKS2Gauss()
{
  if (copymode) return;

  memory->destroy(s_hist_X);
  memory->destroy(s_hist_last);

  FixACKS2Gauss::deallocate_storage();
  FixACKS2Gauss::deallocate_matrix();
}

/* ---------------------------------------------------------------------- */

void FixACKS2Gauss::post_constructor()
{
  memory->create(s_hist_last,2,nprev,"acks2/gauss:s_hist_last");
  for (int i = 0; i < 2; i++)
    for (int j = 0; j < nprev; ++j)
      s_hist_last[i][j] = 0.0;

  grow_arrays(atom->nmax);
  for (int i = 0; i < atom->nmax; i++)
    for (int j = 0; j < nprev; ++j)
      s_hist[i][j] = s_hist_X[i][j] = 0.0;

  pertype_parameters(pertype_option);
  if (dual_enabled)
    error->all(FLERR,"Dual keyword only supported with fix acks2/gauss/omp");
}

/* ---------------------------------------------------------------------- */

void FixACKS2Gauss::pertype_parameters(char *arg)
{
  const int ntypes = atom->ntypes;

  memory->create(chi,ntypes+1,"acsk2/gauss:chi");
  memory->create(eta,ntypes+1,"acsk2/gauss:eta");
  memory->create(zeta,ntypes+1,"acks2/gauss:zeta");
  memory->create(zcore,ntypes+1,"acks2/gauss:zcore");
  memory->create(Xij,(ntypes+1)*(ntypes+1)*4,"acks2/gauss:Xij");

  if (comm->me == 0) {
    chi[0] = eta[0] = zeta[0] = zcore[0] = 0.0;
    for (int n = 0; n <= 3; n++)
      for (int i = 0; i <= ntypes; i++)
        for (int j = 0; j <= ntypes; j++)
          Xij[i*(ntypes+1)*4+j*4+n] = Xij[j*(ntypes+1)*4+i*4+n] = 0.0;
    try {
      TextFileReader reader(arg,"acks2/gauss parameter");
      reader.ignore_comments = false;

      const char *line;
      int i = 1;
      while (i <= ntypes) {
        line = reader.next_line();
        if (!line)
          throw TokenizerException("Fix acks2/gauss: Incorrect parameter file format","");
        ValueTokenizer values(line);

        if (values.count() != 5)
          throw TokenizerException("Fix acks2/gauss: Incorrect parameter file format","");

        int itype = values.next_int();
        if ((itype < 1) || (itype > ntypes))
          throw TokenizerException("Fix acks2/gauss: invalid atom type in parameter file",
                                   std::to_string(itype));

        chi[itype] = values.next_double();
        eta[itype] = values.next_double();
        zeta[itype] = values.next_double();
        zcore[itype] = values.next_double();
        i++;
      }
      while((line = reader.next_line())) {
        ValueTokenizer values(line);
        int itype = values.next_int();
        int jtype = values.next_int();

        if ((itype < 1) || (itype > ntypes))
          throw TokenizerException("Fix acks2/gauss: invalid atom type in parameter file",
                                   std::to_string(itype));
        if ((jtype < 1) || (jtype > ntypes))
          throw TokenizerException("Fix acks2/gauss: invalid atom type in parameter file",
                                   std::to_string(itype));

        for (int n = 0; n <= 3; n++)
          Xij[itype*(ntypes+1)*4+jtype*4+n] = Xij[jtype*(ntypes+1)*4+itype*4+n] = values.next_double();
      }
    } catch (std::exception &e) {
      error->one(FLERR,e.what());
    }
  }

  MPI_Bcast(chi,ntypes+1,MPI_DOUBLE,0,world);
  MPI_Bcast(eta,ntypes+1,MPI_DOUBLE,0,world);
  MPI_Bcast(zeta,ntypes+1,MPI_DOUBLE,0,world);
  MPI_Bcast(zcore,ntypes+1,MPI_DOUBLE,0,world);
  MPI_Bcast(Xij,(ntypes+1)*(ntypes+1)*4,MPI_DOUBLE,0,world);
}

/* ---------------------------------------------------------------------- */

void FixACKS2Gauss::allocate_storage()
{
  nmax = atom->nmax;
  NN = atom->nlocal + atom->nghost;
  const int size = nmax*2 + 2;

  // 0 to nn-1: owned atoms related to H matrix
  // nn to NN-1: ghost atoms related to H matrix
  // NN to NN+nn-1: owned atoms related to X matrix
  // NN+nn to 2*NN-1: ghost atoms related X matrix
  // 2*NN to 2*NN+1: last two rows, owned by proc 0

  memory->create(s,size,"acks2:s");
  memory->create(b_s,size,"acks2:b_s");

  memory->create(Hdia_inv,nmax,"acks2:Hdia_inv");
  memory->create(chi_field,nmax,"acks2:chi_field");

  memory->create(unwrap,atom->nmax,3,"acks2:unwrap");
  memory->create(chi_dfield,nmax,"acks2:chi_dfield");
  memory->create(H_dfield,nmax,nmax,"acks2:H_dfield");
  memory->create(H_dfield_jarray,nmax,nmax,"acks2:H_dfield_jarray");

  memory->create(X_diag,nmax,"acks2:X_diag");
  memory->create(Xdia_inv,nmax,"acks2:Xdia_inv");

  memory->create(p,size,"acks2:p");
  memory->create(q,size,"acks2:q");
  memory->create(r,size,"acks2:r");
  memory->create(d,size,"acks2:d");

  memory->create(g,size,"acks2:g");
  memory->create(q_hat,size,"acks2:q_hat");
  memory->create(r_hat,size,"acks2:r_hat");
  memory->create(y,size,"acks2:y");
  memory->create(z,size,"acks2:z");
  memory->create(u,size,"acks2:u");

  memory->create(special_local, nmax, 4, "acks2:special_local");
  memory->create(nspecial_local, nmax, "acks2:nspecial_local");
}

/* ---------------------------------------------------------------------- */

void FixACKS2Gauss::deallocate_storage()
{
  FixQEqGauss::deallocate_storage();

  memory->destroy(X_diag);
  memory->destroy(Xdia_inv);

  memory->destroy(g);
  memory->destroy(q_hat);
  memory->destroy(r_hat);
  memory->destroy(y);
  memory->destroy(z);
  memory->destroy(u);
  memory->destroy(special_local);
  memory->destroy(nspecial_local);
}

/* ---------------------------------------------------------------------- */

void FixACKS2Gauss::allocate_matrix()
{
  FixQEqGauss::allocate_matrix();

  X.n = n_cap;
  X.m = m_cap;
  memory->create(X.firstnbr,n_cap,"acks2:X.firstnbr");
  memory->create(X.numnbrs,n_cap,"acks2:X.numnbrs");
  memory->create(X.jlist,m_cap,"acks2:X.jlist");
  memory->create(X.val,m_cap,"acks2:X.val");
}

/* ---------------------------------------------------------------------- */

void FixACKS2Gauss::deallocate_matrix()
{
  FixQEqGauss::deallocate_matrix();

  memory->destroy(X.firstnbr);
  memory->destroy(X.numnbrs);
  memory->destroy(X.jlist);
  memory->destroy(X.val);
}

/* ---------------------------------------------------------------------- */

void FixACKS2Gauss::init()
{
  FixQEqGauss::init();
}

/* ---------------------------------------------------------------------- */

void FixACKS2Gauss::init_storage()
{
  if (efield) get_chi_field();
  else if (dfield){
    update_displacements();
    get_chi_dfield();
    compute_H_dfield();
  }

  //for (int ii = 0; ii < NN; ii++) {
  for (int ii = 0; ii < nn; ii++) {
    int i = ilist[ii];
    if (atom->mask[i] & groupbit) {
      b_s[i] = -chi[atom->type[i]];
      if (efield) b_s[i] -= chi_field[i];
      else if (dfield) b_s[i] -= chi_dfield[i];
      b_s[NN + i] = 0.0;
      s[i] = 0.0;
      s[NN + i] = 0.0;
    }
  }

  for (int i = 0; i < 2; i++) {
    b_s[2*NN + i] = 0.0;
    s[2*NN + i] = 0.0;
  }
}

/* ---------------------------------------------------------------------- */

void FixACKS2Gauss::pre_force(int /*vflag*/)
{
  if (update->ntimestep % nevery) return;

  NN = atom->nlocal + atom->nghost;

  nn = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  // grow arrays if necessary
  // need to be atom->nmax in length

  if (atom->nmax > nmax) reallocate_storage();
  if (atom->nlocal > n_cap*DANGER_ZONE || m_fill > m_cap*DANGER_ZONE)
    reallocate_matrix();

  if (efield) get_chi_field();
  else if (dfield) {
    update_displacements();
    get_chi_dfield();
    compute_H_dfield();
  }

  init_matvec();

  matvecs = BiCGStab(b_s, s); // BiCGStab on s - parallel

  calculate_Q();
}

/* ---------------------------------------------------------------------- */

void FixACKS2Gauss::init_matvec()
{
  /* fill-in H matrix */
  compute_H();

  /* fill-in X matrix */
  compute_X();
  pack_flag = 4;
  comm->reverse_comm(this); //Coll_Vector(X_diag);

  int ii, i;

  for (ii = 0; ii < nn; ++ii) {
    if (X_diag[ii] == 0.0)
      Xdia_inv[ii] = 1.0;
    else
      Xdia_inv[ii] = 1.0 / X_diag[ii];

    i = ilist[ii];
    if (atom->mask[i] & groupbit) {

      /* init pre-conditioner for H and init solution vectors */
      if (dfield) Hdia_inv[i] = 1. / (eta[atom->type[i]] + calculate_H_dfield(i,i));
      else Hdia_inv[i] = 1. / eta[atom->type[i]];
      b_s[i] = -chi[atom->type[i]];
      if (efield) b_s[i] -= chi_field[i];
      else if (dfield) b_s[i] -= chi_dfield[i];
      b_s[NN+i] = 0.0;

      /* cubic extrapolation for s from previous solutions */
      s[i] = 4*(s_hist[i][0]+s_hist[i][2])-(6*s_hist[i][1]+s_hist[i][3]);
      s[NN+i] = 4*(s_hist_X[i][0]+s_hist_X[i][2])-(6*s_hist_X[i][1]+s_hist_X[i][3]);
    }
  }

  // last two rows
  if (last_rows_flag) {
    for (i = 0; i < 2; i++) {
      b_s[2*NN+i] = 0.0;
      s[2*NN+i] = 4*(s_hist_last[i][0]+s_hist_last[i][2])-(6*s_hist_last[i][1]+s_hist_last[i][3]);
    }
  }

  pack_flag = 2;
  comm->forward_comm(this); //Dist_vector(s);
  more_forward_comm(s);
}

/* ---------------------------------------------------------------------- */

void FixACKS2Gauss::compute_X()
{
  int jnum;
  int i, j, ii, jj, itype, jtype, moli, molj, flag, intra_flag;
  double dx, dy, dz, r_sqr, c1, c2, X_val;
  const int ntypes = atom->ntypes;
  const double SMALL = 0.0001;

  int *type = atom->type;
  tagint *tag = atom->tag;
  double **x = atom->x;
  int *mask = atom->mask;
  
  // BFJ: stolen from pair_mesocnt
  int **nspecial = atom->nspecial;
  pack_flag = 4;
  comm->forward_comm(this);
  // BFJ: I get segfaults if I don't do these separately
  tagint **special = atom->special;
  pack_flag = 5;
  comm->forward_comm(this);
  
  memory->grow(special_local, nmax, 4, "acks2:special_local");
  memory->grow(nspecial_local, nmax, "acks2:nspecial_local");
  // create version of atom->special with local ids and correct images
  int atomj;
  for (int i = 0; i < atom->nlocal + atom->nghost; i++) {
      nspecial_local[i] = nspecial[i][0];
      for (int j = 0 ; j < nspecial[i][0]; j++) {
        atomj = atom->map(special[i][j]);
        special_local[i][j] = domain->closest_image(i, atomj);
      }
  }
  memset(X_diag,0,atom->nmax*sizeof(double));

  // fill in the X matrix
  m_fill = 0;
  r_sqr = 0;
  for (ii = 0; ii < nn; ii++) {
    i = ilist[ii];
    itype = type[i];
    moli = atom->molecule[i];
    if (mask[i] & groupbit) {
      jlist = firstneigh[i];
      jnum = numneigh[i];
      X.firstnbr[i] = m_fill;

      for (jj = 0; jj < jnum; jj++) {
        j = jlist[jj];
        j &= NEIGHMASK;
        jtype = type[j];
        molj = atom->molecule[j];

        dx = x[j][0] - x[i][0];
        dy = x[j][1] - x[i][1];
        dz = x[j][2] - x[i][2];
        r_sqr = SQR(dx) + SQR(dy) + SQR(dz);

        flag = 0;
        if (r_sqr <= cutoff_sq) {
          if (j < atom->nlocal) flag = 1;
          else if (tag[i] < tag[j]) flag = 1;
          else if (tag[i] == tag[j]) {
            if (dz > SMALL) flag = 1;
            else if (fabs(dz) < SMALL) {
              if (dy > SMALL) flag = 1;
              else if (fabs(dy) < SMALL && dx > SMALL)
                flag = 1;
            }
          }
        }

        if (flag) {
          X.jlist[m_fill] = j;
          intra_flag = 0;
          // BFJ: This is robust
          // and horrible
          // and it limits bonded parametrization to 1-2, 1-3 and 1-4 pairs
          // but at least it's robust
          if (moli == molj) {
            for (int k = 0 ; k < nspecial_local[j]; k++) {
              int atom_jk = special_local[j][k];
              if (atom_jk == i) {
                intra_flag = 1;
                break;
              }
              for (int l = 0 ; l < nspecial_local[i]; l++) {
                if (special_local[i][l] == atom_jk) {
                  intra_flag = 1;
                  break;
                }
              }
              for (int l = 0 ; l < nspecial_local[atom_jk]; l++) {
                int atom_kl = special_local[atom_jk][l];
                if (atom_kl == j) continue;
                if (atom_kl == i) {
                  intra_flag = 1;
                  break;
                }
                for (int m = 0 ; m < nspecial_local[i]; m++) {
                  if (special_local[i][m] == atom_kl) {
                    intra_flag = 1;
                    break;
                  }
                }
              }
            }
          }
          if (intra_flag) {
            c1 = Xij[itype*(ntypes+1)*4+jtype*4+0];
            c2 = Xij[itype*(ntypes+1)*4+jtype*4+1];
            X_val = calculate_X_bonded(sqrt(r_sqr), c1, c2);
          }
          else {
            c1 = Xij[itype*(ntypes+1)*4+jtype*4+2];
            c2 = Xij[itype*(ntypes+1)*4+jtype*4+3];
            X_val = calculate_X_nonbonded(sqrt(r_sqr), c1, c2);
          }
          X.val[m_fill] = X_val;
          X_diag[i] -= X_val;
          X_diag[j] -= X_val;
          m_fill++;
        }
      }

      X.numnbrs[i] = m_fill - X.firstnbr[i];
    }
  }

  if (m_fill >= X.m)
    error->all(FLERR,"Fix acks2/gauss has insufficient ACKS2 X matrix size: m_fill={} X.m={}\n",m_fill,X.m);
}

/* ---------------------------------------------------------------------- */

double FixACKS2Gauss::calculate_X_bonded(double r, double c1, double c2)
{
  return c1+c2*r;
}

/* ---------------------------------------------------------------------- */

double FixACKS2Gauss::calculate_X_nonbonded(double r, double c1, double c2)
{
  return c1*(exp(-c2*r)-(1-c2*(r-cutoff))*exp(-c2*cutoff));
}

/* ---------------------------------------------------------------------- */

int FixACKS2Gauss::BiCGStab(double *b, double *x)
{
  int  i, j;
  double tmp, alpha, beta, omega, sigma, rho, rho_old, rnorm, bnorm;

  int jj;

  sparse_matvec_acks2(&H, &X, x, d);
  pack_flag = 1;
  comm->reverse_comm(this); //Coll_Vector(d);
  more_reverse_comm(d);

  vector_sum(r , 1.,  b, -1., d, nn);
  bnorm = parallel_norm(b, nn);
  rnorm = parallel_norm(r, nn);

  if (bnorm == 0.0) bnorm = 1.0;
  vector_copy(r_hat, r, nn);
  omega = 1.0;
  rho = 1.0;

  for (i = 1; i < imax && rnorm / bnorm > tolerance; ++i) {
    rho = parallel_dot(r_hat, r, nn);
    if (rho == 0.0) break;

    if (i > 1) {
      beta = (rho / rho_old) * (alpha / omega);
      vector_sum(q , 1., p, -omega, z, nn);
      vector_sum(p , 1., r, beta, q, nn);
    } else {
      vector_copy(p, r, nn);
    }

    // pre-conditioning
    for (jj = 0; jj < nn; ++jj) {
      j = ilist[jj];
      if (atom->mask[j] & groupbit) {
        d[j] = p[j] * Hdia_inv[j];
        d[NN+j] = p[NN+j] * Xdia_inv[j];
      }
    }
    // last two rows
    if (last_rows_flag) {
      d[2*NN] = p[2*NN];
      d[2*NN + 1] = p[2*NN + 1];
    }

    pack_flag = 1;
    comm->forward_comm(this); //Dist_vector(d);
    more_forward_comm(d);
    sparse_matvec_acks2(&H, &X, d, z);
    pack_flag = 2;
    comm->reverse_comm(this); //Coll_vector(z);
    more_reverse_comm(z);

    tmp = parallel_dot(r_hat, z, nn);
    alpha = rho / tmp;

    vector_sum(q , 1., r, -alpha, z, nn);

    tmp = parallel_dot(q, q, nn);

    // early convergence check
    if (tmp < tolerance) {
      vector_add(x, alpha, d, nn);
      break;
    }

    // pre-conditioning
    for (jj = 0; jj < nn; ++jj) {
      j = ilist[jj];
      if (atom->mask[j] & groupbit) {
        q_hat[j] = q[j] * Hdia_inv[j];
        q_hat[NN+j] = q[NN+j] * Xdia_inv[j];
      }
    }
    // last two rows
    if (last_rows_flag) {
      q_hat[2*NN] = q[2*NN];
      q_hat[2*NN + 1] = q[2*NN + 1];
    }

    pack_flag = 3;
    comm->forward_comm(this); //Dist_vector(q_hat);
    more_forward_comm(q_hat);
    sparse_matvec_acks2(&H, &X, q_hat, y);
    pack_flag = 3;
    comm->reverse_comm(this); //Dist_vector(y);
    more_reverse_comm(y);

    sigma = parallel_dot(y, q, nn);
    tmp = parallel_dot(y, y, nn);
    omega = sigma / tmp;

    vector_sum(g , alpha, d, omega, q_hat, nn);
    vector_add(x, 1., g, nn);
    vector_sum(r , 1., q, -omega, y, nn);

    rnorm = parallel_norm(r, nn);
    if (omega == 0) break;
    rho_old = rho;
  }

  if (comm->me == 0) {
    if (omega == 0 || rho == 0) {
      error->warning(FLERR,"Fix acks2/gauss BiCGStab numerical breakdown, omega = {:.8}, rho = {:.8}",
                      omega,rho);
    } else if (i >= imax) {
      error->warning(FLERR,"Fix acks2/gauss BiCGStab convergence failed after {} iterations "
                           "at step {}", i, update->ntimestep);
    }
  }

  return i;
}

/* ---------------------------------------------------------------------- */

void FixACKS2Gauss::sparse_matvec_acks2(sparse_matrix *H, sparse_matrix *X, double *x, double *b)
{
  int i, j, itr_j;
  int ii;

  for (ii = 0; ii < nn; ++ii) {
    i = ilist[ii];
    if (atom->mask[i] & groupbit) {
      b[i] = eta[atom->type[i]] * x[i];
      b[NN + i] = X_diag[i] * x[NN + i];
    }
  }

  for (i = atom->nlocal; i < NN; ++i) {
    if (atom->mask[i] & groupbit) {
      b[i] = 0;
      b[NN + i] = 0;
    }
  }
  // last two rows
  b[2*NN] = 0;
  b[2*NN + 1] = 0;

  for (ii = 0; ii < nn; ++ii) {
    i = ilist[ii];
    if (atom->mask[i] & groupbit) {
      // H Matrix
      for (itr_j=H->firstnbr[i]; itr_j<H->firstnbr[i]+H->numnbrs[i]; itr_j++) {
        j = H->jlist[itr_j];
        b[i] += H->val[itr_j] * x[j];
        b[j] += H->val[itr_j] * x[i];
      }

      // X Matrix
      for (itr_j=X->firstnbr[i]; itr_j<X->firstnbr[i]+X->numnbrs[i]; itr_j++) {
        j = X->jlist[itr_j];
        b[NN + i] += X->val[itr_j] * x[NN + j];
        b[NN + j] += X->val[itr_j] * x[NN + i];
      }

      // Identity Matrix
      b[NN + i] += x[i];
      b[i] += x[NN + i];

      // Second-to-last row/column
      b[2*NN] += x[NN + i];
      b[NN + i] += x[2*NN];

      // Last row/column
      b[2*NN + 1] += x[i];
      b[i] += x[2*NN + 1];
    }
  }
  if (dfield){
    int jj;
    for (ii = 0; ii < atom->nlocal; ++ii) {
      i = ilist[ii];
      if (atom->mask[i] & groupbit){
        for (jj = 0; jj < atom->natoms; ++jj){
          j = H_dfield_jarray[i][jj];
          b[i] += H_dfield[i][j] * x[j];
        }
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

void FixACKS2Gauss::calculate_Q()
{
  pack_flag = 2;
  comm->forward_comm(this); //Dist_vector(s);

  for (int i = 0; i < NN; ++i) {
    if (atom->mask[i] & groupbit) {

      atom->q[i] = s[i];
      //BABAK: CHECK SIGN HERE!
      u[i] = -s[NN+i];

      if (i < atom->nlocal) {

        /* backup s */
        for (int k = nprev-1; k > 0; --k) {
          s_hist[i][k] = s_hist[i][k-1];
          s_hist_X[i][k] = s_hist_X[i][k-1];
        }
        s_hist[i][0] = s[i];
        s_hist_X[i][0] = s[NN+i];
      }
    }
  }
  // last two rows
  if (last_rows_flag) {
    for (int i = 0; i < 2; ++i) {
      for (int k = nprev-1; k > 0; --k)
        s_hist_last[i][k] = s_hist_last[i][k-1];
      s_hist_last[i][0] = s[2*NN+i];
    }
  }
}

/* ---------------------------------------------------------------------- */

int FixACKS2Gauss::pack_forward_comm(int n, int *list, double *buf,
                                  int /*pbc_flag*/, int * /*pbc*/)
{
  int m = 0;
  int i, j, k;

  if (pack_flag == 1) {
    for(int i = 0; i < n; i++) {
      int j = list[i];
      buf[m++] = d[j];
      buf[m++] = d[NN+j];
    }
  } else if (pack_flag == 2) {
    for(int i = 0; i < n; i++) {
      int j = list[i];
      buf[m++] = s[j];
      buf[m++] = s[NN+j];
    }
  } else if (pack_flag == 3) {
    for(int i = 0; i < n; i++) {
      int j = list[i];
      buf[m++] = q_hat[j];
      buf[m++] = q_hat[NN+j];
    }
  } else if (pack_flag == 4) {
    //for (i = 0; i < n; i++){
    //  j = list[i];
    //  buf[m++] = ubuf(atom->nspecial[j][0]).d;
    //  buf[m++] = ubuf(atom->special[j][0]).d;
    //  if (atom->nspecial[j][0] == 1) buf[m++] = ubuf(-1).d;
    //  else buf[m++] = ubuf(atom->special[j][1]).d;
    //}
    int **nspecial = atom->nspecial;
    //tagint **special = atom->special;
    for (i = 0; i < n; i++) {
      j = list[i];
      buf[m++] = ubuf(nspecial[j][0]).d;
      //for (k = 0; k < nspecial[j][0]; k++)
      //  buf[m++] = ubuf(special[j][k]).d;
    }
  } else if (pack_flag == 5) {
    int **nspecial = atom->nspecial;
    tagint **special = atom->special;
    for (i = 0; i < n; i++) {
      j = list[i];
      for (k = 0; k < nspecial[j][0]; k++)
        buf[m++] = ubuf(special[j][k]).d;
    }
  }
  return m;
}

/* ---------------------------------------------------------------------- */

void FixACKS2Gauss::unpack_forward_comm(int n, int first, double *buf)
{
  int i, m, k;

  int last = first + n;
  m = 0;

  if (pack_flag == 1) {
    for(i = first; i < last; i++) {
      d[i] = buf[m++];
      d[NN+i] = buf[m++];
    }
  } else if (pack_flag == 2) {
    for(i = first; i < last; i++) {
      s[i] = buf[m++];
      s[NN+i] = buf[m++];
    }
  } else if (pack_flag == 3) {
    for(i = first; i < last; i++) {
      q_hat[i] = buf[m++];
      q_hat[NN+i] = buf[m++];
    }
  } else if (pack_flag == 4) {
    //for (i = first; i < last; i++) {
    //  atom->nspecial[i][0] = (int) ubuf(buf[m++]).i;
    //  atom->special[i][0] = (tagint) ubuf(buf[m++]).i;
    //  if (atom->nspecial[i][0] > 1) atom->special[i][1] = (tagint) ubuf(buf[m]).i;
    //  m++;
    //}
    int **nspecial = atom->nspecial;
    //tagint **special = atom->special;
    for (i = first; i < last; i++) {
      nspecial[i][0] = (int) ubuf(buf[m++]).i;
      //for (k = 0; k < nspecial[i][0]; k++)
      //  special[i][k] = (tagint) ubuf(buf[m++]).i;
    }
  } else if (pack_flag == 5) {
    int **nspecial = atom->nspecial;
    tagint **special = atom->special;
    for (i = first; i < last; i++) {
      for (k = 0; k < nspecial[i][0]; k++)
        special[i][k] = (tagint) ubuf(buf[m++]).i;
    }
  }
}

/* ---------------------------------------------------------------------- */

int FixACKS2Gauss::pack_reverse_comm(int n, int first, double *buf)
{
  int i, m;
  m = 0;
  int last = first + n;

  if (pack_flag == 1) {
    for(i = first; i < last; i++) {
      buf[m++] = d[i];
      buf[m++] = d[NN+i];
    }
  } else if (pack_flag == 2) {
    for(i = first; i < last; i++) {
      buf[m++] = z[i];
      buf[m++] = z[NN+i];
    }
  } else if (pack_flag == 3) {
    for(i = first; i < last; i++) {
      buf[m++] = y[i];
      buf[m++] = y[NN+i];
    }
  } else if (pack_flag == 4) {
    for(i = first; i < last; i++)
      buf[m++] = X_diag[i];
  }

  return m;
}

/* ---------------------------------------------------------------------- */

void FixACKS2Gauss::unpack_reverse_comm(int n, int *list, double *buf)
{
  int j;
  int m = 0;
  if (pack_flag == 1) {
    for(int i = 0; i < n; i++) {
      j = list[i];
      d[j] += buf[m++];
      d[NN+j] += buf[m++];
    }
  } else if (pack_flag == 2) {
    for(int i = 0; i < n; i++) {
      j = list[i];
      z[j] += buf[m++];
      z[NN+j] += buf[m++];
    }
  } else if (pack_flag == 3) {
    for(int i = 0; i < n; i++) {
      j = list[i];
      y[j] += buf[m++];
      y[NN+j] += buf[m++];
    }
  } else if (pack_flag == 4) {
    for(int i = 0; i < n; i++) {
      j = list[i];
      X_diag[j] += buf[m++];
    }
  }
}

/* ----------------------------------------------------------------------
   one proc broadcasts last two rows of vector to everyone else
------------------------------------------------------------------------- */

void FixACKS2Gauss::more_forward_comm(double *vec)
{
  MPI_Bcast(&vec[2*NN],2,MPI_DOUBLE,last_rows_rank,world);
}

/* ----------------------------------------------------------------------
   reduce last two rows of vector and give to one proc
------------------------------------------------------------------------- */

void FixACKS2Gauss::more_reverse_comm(double *vec)
{
  if (last_rows_flag)
    MPI_Reduce(MPI_IN_PLACE,&vec[2*NN],2,MPI_DOUBLE,MPI_SUM,last_rows_rank,world);
  else
    MPI_Reduce(&vec[2*NN],nullptr,2,MPI_DOUBLE,MPI_SUM,last_rows_rank,world);
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based arrays
------------------------------------------------------------------------- */

double FixACKS2Gauss::memory_usage()
{
  double bytes;
  const double size = 2.0*nmax + 2.0;

  bytes = size*nprev * sizeof(double); // s_hist
  bytes += nmax*4.0 * sizeof(double); // storage
  bytes += size*11.0 * sizeof(double); // storage
  bytes += n_cap*4.0 * sizeof(int); // matrix...
  bytes += m_cap*2.0 * sizeof(int);
  bytes += m_cap*2.0 * sizeof(double);

  return bytes;
}

/* ----------------------------------------------------------------------
   allocate solution history array
------------------------------------------------------------------------- */

void FixACKS2Gauss::grow_arrays(int nmax)
{
  memory->grow(s_hist,nmax,nprev,"acks2:s_hist");
  memory->grow(s_hist_X,nmax,nprev,"acks2:s_hist_X");
}

/* ----------------------------------------------------------------------
   copy values within solution history array
------------------------------------------------------------------------- */

void FixACKS2Gauss::copy_arrays(int i, int j, int /*delflag*/)
{
  for (int m = 0; m < nprev; m++) {
    s_hist[j][m] = s_hist[i][m];
    s_hist_X[j][m] = s_hist_X[i][m];
  }
}

/* ----------------------------------------------------------------------
   pack values in local atom-based array for exchange with another proc
------------------------------------------------------------------------- */

int FixACKS2Gauss::pack_exchange(int i, double *buf)
{
  for (int m = 0; m < nprev; m++) buf[m] = s_hist[i][m];
  for (int m = 0; m < nprev; m++) buf[nprev+m] = s_hist_X[i][m];
  return nprev*2;
}

/* ----------------------------------------------------------------------
   unpack values in local atom-based array from exchange with another proc
------------------------------------------------------------------------- */

int FixACKS2Gauss::unpack_exchange(int nlocal, double *buf)
{
  for (int m = 0; m < nprev; m++) s_hist[nlocal][m] = buf[m];
  for (int m = 0; m < nprev; m++) s_hist_X[nlocal][m] = buf[nprev+m];
  return nprev*2;
}

/* ---------------------------------------------------------------------- */

double FixACKS2Gauss::parallel_norm(double *v, int n)
{
  int  i;
  double my_sum, norm_sqr;

  int ii;

  my_sum = 0.0;
  norm_sqr = 0.0;
  for (ii = 0; ii < n; ++ii) {
    i = ilist[ii];
    if (atom->mask[i] & groupbit) {
      my_sum += SQR(v[i]);
      my_sum += SQR(v[NN+i]);
    }
  }

  // last two rows
  if (last_rows_flag) {
    my_sum += SQR(v[2*NN]);
    my_sum += SQR(v[2*NN + 1]);
  }

  MPI_Allreduce(&my_sum, &norm_sqr, 1, MPI_DOUBLE, MPI_SUM, world);

  return sqrt(norm_sqr);
}

/* ---------------------------------------------------------------------- */

double FixACKS2Gauss::parallel_dot(double *v1, double *v2, int n)
{
  int  i;
  double my_dot, res;

  int ii;

  my_dot = 0.0;
  res = 0.0;
  for (ii = 0; ii < n; ++ii) {
    i = ilist[ii];
    if (atom->mask[i] & groupbit) {
      my_dot += v1[i] * v2[i];
      my_dot += v1[NN+i] * v2[NN+i];
    }
  }

  // last two rows
  if (last_rows_flag) {
    my_dot += v1[2*NN] * v2[2*NN];
    my_dot += v1[2*NN + 1] * v2[2*NN + 1];
  }

  MPI_Allreduce(&my_dot, &res, 1, MPI_DOUBLE, MPI_SUM, world);

  return res;
}

/* ---------------------------------------------------------------------- */

double FixACKS2Gauss::parallel_vector_acc(double *v, int n)
{
  int  i;
  double my_acc, res;

  int ii;

  my_acc = 0.0;
  res = 0.0;
  for (ii = 0; ii < n; ++ii) {
    i = ilist[ii];
    if (atom->mask[i] & groupbit) {
      my_acc += v[i];
      my_acc += v[NN+i];
    }
  }

  // last two rows
  if (last_rows_flag) {
    my_acc += v[2*NN];
    my_acc += v[2*NN + 1];
  }

  MPI_Allreduce(&my_acc, &res, 1, MPI_DOUBLE, MPI_SUM, world);

  return res;
}

/* ---------------------------------------------------------------------- */

void FixACKS2Gauss::vector_sum(double* dest, double c, double* v,
                                double d, double* y, int k)
{
  int kk;

  for (--k; k>=0; --k) {
    kk = ilist[k];
    if (atom->mask[kk] & groupbit) {
      dest[kk] = c * v[kk] + d * y[kk];
      dest[NN + kk] = c * v[NN + kk] + d * y[NN + kk];
    }
  }

  // last two rows
  if (last_rows_flag) {
    dest[2*NN] = c * v[2*NN] + d * y[2*NN];
    dest[2*NN + 1] = c * v[2*NN + 1] + d * y[2*NN + 1];
  }
}

/* ---------------------------------------------------------------------- */

void FixACKS2Gauss::vector_add(double* dest, double c, double* v, int k)
{
  int kk;

  for (--k; k>=0; --k) {
    kk = ilist[k];
    if (atom->mask[kk] & groupbit) {
      dest[kk] += c * v[kk];
      dest[NN + kk] += c * v[NN + kk];
    }
  }

  // last two rows
  if (last_rows_flag) {
    dest[2*NN] += c * v[2*NN];
    dest[2*NN + 1] += c * v[2*NN + 1];
  }
}


/* ---------------------------------------------------------------------- */

void FixACKS2Gauss::vector_copy(double* dest, double* v, int k)
{
  int kk;

  for (--k; k>=0; --k) {
    kk = ilist[k];
    if (atom->mask[kk] & groupbit) {
      dest[kk] = v[kk];
      dest[NN + kk] = v[NN + kk];
    }
  }

  // last two rows
  if (last_rows_flag) {
    dest[2*NN] = v[2*NN];
    dest[2*NN + 1] = v[2*NN + 1];
  }
}

