// clang-format off
/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Tod A Pascal (Caltech)
------------------------------------------------------------------------- */

#include "pair_manybody_donor_acceptor_sfg.h"

#include "atom.h"
#include "atom_vec.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "math_const.h"
#include "math_special.h"
#include "memory.h"
#include "molecule.h"
#include "neigh_list.h"
#include "neighbor.h"

#include <cmath>
#include <cstring>

using namespace LAMMPS_NS;
using namespace MathConst;
using namespace MathSpecial;

#define SMALL 0.001
#define CHUNK 9

/* ---------------------------------------------------------------------- */

PairManybodyDonorAcceptorSFG::PairManybodyDonorAcceptorSFG(LAMMPS *lmp) : Pair(lmp)
{
  // hbond cannot compute virial as F dot r
  // due to using map() to find bonded H atoms which are not near donor atom

  no_virial_fdotr_compute = 1;
  restartinfo = 0;

  nparams = maxparam = 0;
  params = nullptr;

  nextra = 2;
  pvector = new double[2];
}

/* ---------------------------------------------------------------------- */

PairManybodyDonorAcceptorSFG::~PairManybodyDonorAcceptorSFG()
{
  memory->sfree(params);
  delete [] pvector;

  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);

    // delete [] donor;
    // delete [] acceptor;
    memory->destroy(type2param);
  }
}

/* ---------------------------------------------------------------------- */

void PairManybodyDonorAcceptorSFG::compute(int eflag, int vflag)
{
  int i,j,m,ii,jj,inum,jnum,itype,jtype,iatom,imol;
  tagint tagprev;
  double delx,dely,delz,rsq,r_DA_sq,rsq2,d_DA,r2;
  double factor_hb,force_angle,force_kernel,evdwl,eng_lj,ehbond,force_switch;
  double c,s,a,b,ac,a11,a12,a22,vx1,vx2,vy1,vy2,vz1,vz2,d;
  double fj[3],fc1[3],fc2[3],r_DA[3],delr2[3],r_DC1[3],r_DC2[3];
  double r2inv,r10inv;
  double switch1,switch2;
  int *ilist,*jlist,*numneigh,**firstneigh;

  double norm_da, norm_dcc, norm_cc_cross, alpha_dot, beta_dot, cut;
  double r_DCC[3], cc_cross[3];
  double D_pi_beta, E_pi, D_sig_alpha, E_sig;
  double dE_sig_dr, dE_pi_dr, dE_dr, dE_sig_dadot, dE_sig_dbdot, dE_pi_dadot, dE_pi_dbdot;
  double dE_dadot, dE_dbdot, cross_c11_c21, cross_c02_20, a_dot_a, sqrt_a_dot_a;
  double dadx_term1, dadx_term2, dbdx_term1, dbdx_term2, dbdx_terma, dbdx_termc;
  double morse_pi_r, morse_sig_r;
  double dr_dx, da_dx, db_dx, morse_pi_cut, morse_sig_cut;
  double exp_pi_r, exp_pi_cut, exp_sig_r, exp_sig_cut;
  double exp_pi_m1_r, exp_pi_m1_cut, exp_sig_m1_r, exp_sig_m1_cut, pre_pi, pre_sig;
  double E_r, E_cut, dE_dr_cut, d2E_dr2_cut, E_sfg, dE_dr_r, dE_sfg_dr;
  double dpre_sig_da, dE_r_da, dE_cut_da, dE_dr_cut_da, d2E_dr2_cut_da, dE_sfg_da;
  double dpre_pi_db, dE_r_db, dE_cut_db, dE_dr_cut_db, d2E_dr2_cut_db, dE_sfg_db;
  double e_test;
  // tagint *klist;

  e_test = 0.0;
  ev_init(eflag,vflag);

  double **x = atom->x;
  double **f = atom->f;
  tagint *tag = atom->tag;
  int *molindex = atom->molindex;
  int *molatom = atom->molatom;
  tagint **special = atom->special;
  int **nspecial = atom->nspecial;
  int *type = atom->type;
  double *special_lj = force->special_lj;
  int molecular = atom->molecular;
  Molecule **onemols = atom->avec->onemols;

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  // printf("\t\t\t\tCompute\n");
  // printf("XYZ:\n");
  // for (i = 0; i < 8; i++) {
  //   printf("%12.8f %12.8f %12.8f\n",x[i][0], x[i][1], x[i][2]);
  // }
  // printf("DONE\n");

  // // // ii = loop over donors
  // // // jj = loop over acceptors
  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    itype = type[i];
    // printf("ii: %i, i: %i, itype: %i\n", ii, i, itype);
    if (donor_typeid != itype) {
      // printf("skipping, atom %i is of type %i which is not the donor type %i\n", i, itype, donor_typeid);
      continue;
    }
    jlist = firstneigh[i];
    jnum = numneigh[i];
    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      factor_hb = special_lj[sbmask(j)];
      j &= NEIGHMASK;

      jtype = type[j];
      // printf("\tjj: %i, j: %i, jtype: %i\n", j, j, jtype);
      if (acceptor_typeid != jtype) {
        // printf("\tskipping, atom %i is of type %i which is not the acceptor type %i\n", i, itype, acceptor_typeid);
        continue;
      }
      // printf("%12.8f %12.8f %12.8f\n",x[i][0], x[i][1], x[i][2]);
      // printf("%12.8f %12.8f %12.8f\n",x[i][0], x[i][1], x[i][2]);
      delx = x[i][0] - x[j][0];
      dely = x[i][1] - x[j][1];
      delz = x[i][2] - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz; 

      m = type2param[itype][jtype];
      if (m < 0) continue;
      const Param &pm = params[m];
      double dp_0 = pm.dp_0;
      double ds_0 = pm.ds_0;
      double a_p  = pm.a_p;
      double a_s  = pm.a_s;
      double rp_0 = pm.rp_0;
      double rs_0 = pm.rs_0;

      int c1_id, c2_id;
      c1_id = atom->map(special[i][0]);
      c2_id = atom->map(special[i][1]);
      // printf("##########%i, %i\n", c1_id, c2_id);
      if (sqrt(rsq) >= pm.cutoff_dsf) {
        // printf("%f %f\n", sqrt(rsq), pm.cutoff_dsf);
        // printf("cutoff used\n");
        continue;
      }
      // RIC calculation
      r_DA[0] = x[j][0] - x[i][0];
      r_DA[1] = x[j][1] - x[i][1];
      r_DA[2] = x[j][2] - x[i][2];
      domain->minimum_image(r_DA);
      r_DA_sq = r_DA[0]*r_DA[0] + r_DA[1]*r_DA[1] + r_DA[2]*r_DA[2];
      norm_da = sqrt(r_DA_sq);

      r_DC1[0] = x[c1_id][0] - x[i][0];
      r_DC1[1] = x[c1_id][1] - x[i][1];
      r_DC1[2] = x[c1_id][2] - x[i][2];
      domain->minimum_image(r_DC1);

      r_DC2[0] = x[c2_id][0] - x[i][0];
      r_DC2[1] = x[c2_id][1] - x[i][1];
      r_DC2[2] = x[c2_id][2] - x[i][2];
      domain->minimum_image(r_DC2);

      r_DCC[0] = - (r_DC1[0] + r_DC2[0]);
      r_DCC[1] = - (r_DC1[1] + r_DC2[1]);
      r_DCC[2] = - (r_DC1[2] + r_DC2[2]);
      norm_dcc = sqrt(r_DCC[0]*r_DCC[0]+r_DCC[1]*r_DCC[1]+r_DCC[2]*r_DCC[2]);

      cc_cross[0] = r_DC1[1]*r_DC2[2] - r_DC1[2]*r_DC2[1];
      cc_cross[1] = r_DC1[2]*r_DC2[0] - r_DC1[0]*r_DC2[2];
      cc_cross[2] = r_DC1[0]*r_DC2[1] - r_DC1[1]*r_DC2[0];
      norm_cc_cross = sqrt(cc_cross[0]*cc_cross[0] + cc_cross[1]*cc_cross[1] + cc_cross[2]*cc_cross[2]);
      // printf("DEBUG VECTORS: %12.8f %12.8f %12.8f\n", x[i][0], x[i][1], x[i][2]);
      // printf("DEBUG VECTORS: %12.8f %12.8f %12.8f\n", r_DA[0], r_DA[1], r_DA[2]);
      // printf("DEBUG VECTORS: %12.8f %12.8f %12.8f\n", r_DC1[0], r_DC1[1], r_DC1[2]);
      // printf("DEBUG VECTORS: %12.8f %12.8f %12.8f\n", r_DC2[0], r_DC2[1], r_DC2[2]);
      alpha_dot = (r_DCC[0]*r_DA[0]+r_DCC[1]*r_DA[1]+r_DCC[2]*r_DA[2])/(norm_da * norm_dcc);
      beta_dot = (cc_cross[0]*r_DA[0]+cc_cross[1]*r_DA[1]+cc_cross[2]*r_DA[2])/(norm_da * norm_cc_cross);
      // printf("a:%12.8f b:%12.8f\n", alpha_dot, beta_dot);
      cut = pm.cutoff_dsf;
      // printf("%f %f %f\n", norm_da, beta_dot, alpha_dot);
      // Energy calc
      exp_pi_r = exp(-a_p * (norm_da - rp_0)); // "works"
      exp_pi_cut = exp(-a_p * (cut - rp_0));
      exp_sig_r = exp(-a_s * (norm_da - rs_0)); // DOES NOT WORK
      exp_sig_cut = exp(-a_s * (cut - rs_0));
      D_pi_beta = beta_dot*beta_dot;
      D_sig_alpha = (1.5 + alpha_dot)/(2.5) * alpha_dot*alpha_dot;

      exp_pi_m1_r = exp_pi_r - 1;
      exp_pi_m1_cut = exp_pi_cut - 1;
      exp_sig_m1_r = exp_sig_r - 1;
      exp_sig_m1_cut = exp_sig_cut - 1;
      pre_pi = dp_0 * D_pi_beta;
      pre_sig = ds_0 * D_sig_alpha;
      // printf("%f %f %f %f %f %f\n", exp_pi_m1_r, exp_pi_m1_cut,exp_sig_m1_r,  exp_sig_m1_r, pre_pi, pre_sig);

      E_r =  pre_sig * exp_sig_m1_r*exp_sig_m1_r + 
             pre_pi  * exp_pi_m1_r*exp_pi_m1_r;
      printf("DEBUG: %f %f %f %f\n", pre_sig, exp_sig_m1_r, pre_pi, exp_pi_m1_r);
      E_cut =  pre_sig * exp_sig_m1_cut*exp_sig_m1_cut + 
               pre_pi  * exp_pi_m1_cut*exp_pi_m1_cut;
      dE_dr_cut = -2 * (pre_sig * a_s * exp_sig_cut * exp_sig_m1_cut + pre_pi * a_p * exp_pi_cut * exp_pi_m1_cut);
      d2E_dr2_cut = 2 * (pre_sig * a_s*a_s * exp_sig_cut * (2*exp_sig_cut - 1) + 
                         pre_pi  * a_p*a_p * exp_pi_cut  * (2*exp_pi_cut  - 1));
      E_sfg = E_r - E_cut - (norm_da - cut)*dE_dr_cut - 0.5*(norm_da - cut)*(norm_da - cut)*d2E_dr2_cut;
      e_test = E_sfg;
      printf("DEBUG - energy: %12.8f\n", E_r);
      printf("DEBUG - dist: %12.8f\n", norm_da);

      // Force calc
      dE_dr_r = -2 * (pre_sig * a_s * exp_sig_r * exp_sig_m1_r + pre_pi * a_p * exp_pi_r * exp_pi_m1_r);
      dE_sfg_dr = dE_dr_r - dE_dr_cut - (norm_da - cut)*d2E_dr2_cut;
      dE_dr = dE_sfg_dr;
      
      dpre_sig_da = ds_0 * alpha_dot*(1.2*alpha_dot+1.2);
      dE_r_da =  dpre_sig_da * exp_sig_m1_r*exp_sig_m1_r;
      dE_cut_da =  dpre_sig_da * exp_sig_m1_cut*exp_sig_m1_cut;
      dE_dr_cut_da = -2 * dpre_sig_da * a_s * exp_sig_cut * exp_sig_m1_cut;
      d2E_dr2_cut_da = 2 * dpre_sig_da * a_s*a_s * exp_sig_cut * (2*exp_sig_cut - 1);
      dE_sfg_da = dE_r_da - dE_cut_da - (norm_da - cut)*dE_dr_cut_da - 0.5*(norm_da - cut)*(norm_da - cut)*d2E_dr2_cut_da;
      dE_dadot = dE_sfg_da;

      dpre_pi_db = dp_0 * 2*beta_dot;
      dE_r_db =  dpre_pi_db * exp_pi_m1_r*exp_pi_m1_r;
      dE_cut_db =  dpre_pi_db * exp_pi_m1_cut*exp_pi_m1_cut;
      dE_dr_cut_db = -2 * dpre_pi_db * a_p * exp_pi_cut * exp_pi_m1_cut;
      d2E_dr2_cut_db = 2 * dpre_pi_db * a_p*a_p * exp_pi_cut * (2*exp_pi_cut - 1);
      dE_sfg_db = dE_r_db - dE_cut_db - (norm_da - cut)*dE_dr_cut_db - 0.5*(norm_da - cut)*(norm_da - cut)*d2E_dr2_cut_db;
      dE_dbdot = dE_sfg_db;

      cross_c11_c21 = (r_DC1[1]*r_DC2[2] - r_DC1[2]*r_DC2[1]);
      cross_c02_20 = (r_DC1[0]*r_DC2[1] - r_DC1[1]*r_DC2[0]);

      a_dot_a = r_DA[0]*r_DA[0] + r_DA[1]*r_DA[1] + r_DA[2]*r_DA[2];
      sqrt_a_dot_a = sqrt(a_dot_a);

      dadx_term1 = (r_DA[0]*(r_DC1[0] + r_DC2[0]) + r_DA[1]*(r_DC1[1] + r_DC2[1]) + r_DA[2]*(r_DC1[2] + r_DC2[2]));
      dadx_term2 = pow(r_DC1[0] + r_DC2[0], 2.0) + pow(r_DC1[1] + r_DC2[1], 2.0) + pow(r_DC1[2] + r_DC2[2], 2.0);

      dbdx_term1 = (r_DA[0]*cross_c11_c21 - r_DA[1]*(r_DC1[0]*r_DC2[2] - r_DC1[2]*r_DC2[0]) + r_DA[2]*cross_c02_20);
      dbdx_term2 = (cross_c02_20*cross_c02_20 + pow(r_DC1[0]*r_DC2[2] - r_DC1[2]*r_DC2[0], 2.0) + cross_c11_c21*cross_c11_c21);
      dbdx_terma = (pow(a_dot_a, 3.0/2.0)*sqrt(dbdx_term2));
      // printf("dbdx_term2, a_dot_a, dbdx_terma: %12.8f %12.8f %12.8f\n", dbdx_term2, a_dot_a, dbdx_terma);
      dbdx_termc = (sqrt_a_dot_a*pow(dbdx_term2, 3.0/2.0));
    
    
      for (int k = 0; k < 3; k++) {
        // # acceptor
        dr_dx = r_DA[k]/sqrt_a_dot_a;
        da_dx = (r_DA[k]*dadx_term1 - (r_DC1[k] + r_DC2[k])*(a_dot_a))/(pow(a_dot_a, 3.0/2.0)*sqrt(dadx_term2));
          
        if (k == 0) {
          db_dx = (-r_DA[0]*dbdx_term1 + cross_c11_c21*(a_dot_a))/dbdx_terma;
        } else if (k == 1) {
          db_dx = (-r_DA[1]*dbdx_term1 + (-r_DC1[0]*r_DC2[2] + r_DC1[2]*r_DC2[0])*(a_dot_a))/dbdx_terma;
        } else if (k == 2) {
          db_dx = (-r_DA[2]*dbdx_term1 + cross_c02_20*(a_dot_a))/dbdx_terma;
        }
        f[j][k] = -(dE_dr * dr_dx + dE_dadot * da_dx + dE_dbdot * db_dx);
        // printf("%12.8f, %12.8f, %12.8f, %12.8f, %12.8f, %12.8f\n", dE_dr, dr_dx, dE_dadot, da_dx,  dE_dbdot, db_dx);
        // printf("%12.8f\n", dE_dr * dr_dx + dE_dadot * da_dx + dE_dbdot * db_dx);
        // #c1
        dr_dx = 0;
        da_dx = (-r_DA[k]*(dadx_term2) + (r_DC1[k] + r_DC2[k])*dadx_term1)/(sqrt_a_dot_a*pow(dadx_term2, 3.0/2.0));
        if (k == 0) {
          db_dx = ((-r_DA[1]*r_DC2[2] + r_DA[2]*r_DC2[1])*dbdx_term2 - (r_DC2[1]*cross_c02_20 + r_DC2[2]*(r_DC1[0]*r_DC2[2] - r_DC1[2]*r_DC2[0]))*dbdx_term1)/dbdx_termc;
        } else if (k == 1) {
          db_dx = ((r_DA[0]*r_DC2[2] - r_DA[2]*r_DC2[0])*dbdx_term2 + (r_DC2[0]*cross_c02_20 - r_DC2[2]*cross_c11_c21)*dbdx_term1)/dbdx_termc;
        } else if (k == 2) {
          db_dx = ((-r_DA[0]*r_DC2[1] + r_DA[1]*r_DC2[0])*dbdx_term2 + (r_DC2[0]*(r_DC1[0]*r_DC2[2] - r_DC1[2]*r_DC2[0]) + r_DC2[1]*cross_c11_c21)*dbdx_term1)/dbdx_termc;
        }
        f[c1_id][k] = -(dE_dr * dr_dx + dE_dadot * da_dx + dE_dbdot * db_dx);

        // #c2
        if (k == 0) {
          db_dx = ((r_DA[1]*r_DC1[2] - r_DA[2]*r_DC1[1])*dbdx_term2 + (r_DC1[1]*cross_c02_20 + r_DC1[2]*(r_DC1[0]*r_DC2[2] - r_DC1[2]*r_DC2[0]))*dbdx_term1)/dbdx_termc;
        } else if (k == 1) {
          db_dx = ((-r_DA[0]*r_DC1[2] + r_DA[2]*r_DC1[0])*dbdx_term2 - (r_DC1[0]*cross_c02_20 - r_DC1[2]*cross_c11_c21)*dbdx_term1)/dbdx_termc;
        } else if (k == 2) {
          db_dx = ((r_DA[0]*r_DC1[1] - r_DA[1]*r_DC1[0])*dbdx_term2 - (r_DC1[0]*(r_DC1[0]*r_DC2[2] - r_DC1[2]*r_DC2[0]) + r_DC1[1]*cross_c11_c21)*dbdx_term1)/dbdx_termc;
        }
        f[c2_id][k] = -(dE_dr * dr_dx + dE_dadot * da_dx + dE_dbdot * db_dx);
    
        // #donor
        f[i][k] = - (f[j][k] + f[c1_id][k] + f[c2_id][k]);
        // printf("i:%i k:% \n", i,k);

      }
        // printf("#########\n");
        // for (int i = 0; i < inum; i++) {
        //   for (int j = 0; j < 3; j++) {
        //     printf("%12.8f ", f[i][j]);
        //   }
        //   printf("\n");
        // }
    fj[0] = f[j][0];
    fj[1] = f[j][1];
    fj[2] = f[j][2];

    fc1[0] = f[c1_id][0];
    fc1[1] = f[c1_id][1];
    fc1[2] = f[c1_id][2];

    fc2[0] = f[c2_id][0];
    fc2[1] = f[c2_id][1];
    fc2[2] = f[c2_id][2];
    if (evflag) ev_tally4(j,c1_id,c2_id,i,e_test,fj,fc1,fc2,r_DA,r_DC1,r_DC2);
    }
  }

}

// double PairManybodyDonorAcceptorSFG::energy_base(double r, double a, double b)
// {
//   return e;
// }

/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

void PairManybodyDonorAcceptorSFG::allocate()
{
  allocated = 1;
  int n = atom->ntypes; // get number of atomtypes

  // mark all setflag as set, since don't require pair_coeff of all I,J

  memory->create(setflag,n+1,n+1,"pair:setflag");
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++)
      setflag[i][j] = 1;

  memory->create(cutsq,n+1,n+1,"pair:cutsq");

  // donor = new int[n+1];
  // acceptor = new int[n+1];
  // memory->create(type2param,n+1,n+1,n+1,"pair:type2param");
  memory->create(type2param,n+1,n+1,"pair:type2param");

  int i,j;
  for (i = 1; i <= n; i++)
    for (j = 1; j <= n; j++)
        type2param[i][j] = -1;
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairManybodyDonorAcceptorSFG::settings(int narg, char **arg)
{
  if (narg != 1) error->all(FLERR,"Wrong number of args given to ManybodyDonorAcceptor");
  cutoff_dsf = utils::numeric(FLERR,arg[0],false,lmp);
  // if (narg != 4) error->all(FLERR,"Illegal pair_style command");

  // ap_global = utils::inumeric(FLERR,arg[0],false,lmp);
  // cut_inner_global = utils::numeric(FLERR,arg[1],false,lmp);
  // cut_outer_global = utils::numeric(FLERR,arg[2],false,lmp);
  // cut_angle_global = utils::numeric(FLERR,arg[3],false,lmp) * MY_PI/180.0;
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairManybodyDonorAcceptorSFG::coeff(int narg, char **arg)
{
  // printf("%d\n",narg);
  if (narg != 9) {
    error->all(FLERR,"Incorrect args for pair coefficients, need 6 (2 ids, 1 donor/acceptor, 3 pars)");
  }
  if (!allocated) allocate();

  // int ilo,ihi,jlo,jhi,klo,khi;
  // utils::bounds(FLERR,arg[0],1,atom->ntypes,ilo,ihi,error);
  // utils::bounds(FLERR,arg[1],1,atom->ntypes,jlo,jhi,error);
  // utils::bounds(FLERR,arg[2],1,atom->ntypes,klo,khi,error);
  int ilo,ihi,jlo,jhi;
  utils::bounds(FLERR,arg[0],1,atom->ntypes,ilo,ihi,error); // arg[0] is always the donor
  utils::bounds(FLERR,arg[1],1,atom->ntypes,jlo,jhi,error); // arg[1] is always the acceptor
  printf("ilo, ihi: %i %i\n", ilo, ihi);
  printf("jlo, jhi: %i %i\n", jlo, jhi);

  int type_i, type_j;
  type_i = utils::numeric(FLERR,arg[0],false,lmp);
  type_j  = utils::numeric(FLERR,arg[1],false,lmp);

  if (strcmp(arg[2],"i") == 0) {
    donor_typeid = MIN(type_i, type_j);
    acceptor_typeid = MAX(type_i, type_j);
  } else if (strcmp(arg[2],"j") == 0) {
    donor_typeid = MAX(type_i, type_j);
    acceptor_typeid = MIN(type_i, type_j);
  } else {
    error->all(FLERR,"Incorrect args for pair coefficients, third flag must be either i or j");
  }
  printf("donortype %i\n", donor_typeid);
  printf("acceptor_typeid %i\n", acceptor_typeid);

  // int donor_flag = 0;
  // if (strcmp(arg[3],"i") == 0) donor_flag = 0;
  // else if (strcmp(arg[3],"j") == 0) donor_flag = 1;
  // else error->all(FLERR,"Incorrect args for pair coefficients");

  // double epsilon_one = utils::numeric(FLERR,arg[4],false,lmp);
  // double sigma_one = utils::numeric(FLERR,arg[5],false,lmp);
  double dp_0 = utils::numeric(FLERR,arg[3],false,lmp);
  double ds_0 = utils::numeric(FLERR,arg[4],false,lmp);
  double a_p  = utils::numeric(FLERR,arg[5],false,lmp);
  double a_s  = utils::numeric(FLERR,arg[6],false,lmp);
  double rp_0 = utils::numeric(FLERR,arg[7],false,lmp);
  double rs_0 = utils::numeric(FLERR,arg[8],false,lmp);
  // double de = utils::numeric(FLERR,arg[3],false,lmp);
  // double a  = utils::numeric(FLERR,arg[4],false,lmp);
  // double r0 = utils::numeric(FLERR,arg[5],false,lmp);

  // int ap_one = ap_global;
  // if (narg > 6) ap_one = utils::inumeric(FLERR,arg[6],false,lmp);
  // double cut_inner_one = cut_inner_global;
  // double cut_outer_one = cut_outer_global;
  // if (narg > 8) {
  //   cut_inner_one = utils::numeric(FLERR,arg[7],false,lmp);
  //   cut_outer_one = utils::numeric(FLERR,arg[8],false,lmp);
  // }
  // if (cut_inner_one>cut_outer_one)
  //   error->all(FLERR,"Pair inner cutoff >= Pair outer cutoff");
  // double cut_angle_one = cut_angle_global;
  // if (narg == 10) cut_angle_one = utils::numeric(FLERR,arg[9],false,lmp) * MY_PI/180.0;
  // grow params array if necessary

  if (nparams == maxparam) {
    maxparam += CHUNK;
    params = (Param *) memory->srealloc(params,maxparam*sizeof(Param),
                                        "pair:params");

    // make certain all addional allocated storage is initialized
    // to avoid false positives when checking with valgrind

    memset(params + nparams, 0, CHUNK*sizeof(Param));
  }

  params[nparams].dp_0 = dp_0;
  params[nparams].ds_0 = ds_0;
  params[nparams].a_p = a_p;
  params[nparams].a_s = a_s;
  params[nparams].rp_0 = rp_0;
  params[nparams].rs_0 = rs_0;
  params[nparams].cutoff_dsf = cutoff_dsf;

  // params[nparams].epsilon = epsilon_one;
  // params[nparams].sigma = sigma_one;
  // params[nparams].ap = ap_one;
  // params[nparams].cut_inner = cut_inner_one;
  // params[nparams].cut_outer = cut_outer_one;
  // params[nparams].cut_innersq = cut_inner_one*cut_inner_one;
  // params[nparams].cut_outersq = cut_outer_one*cut_outer_one;
  // params[nparams].cut_angle = cut_angle_one;
  // params[nparams].denom_vdw =
  //   (params[nparams].cut_outersq-params[nparams].cut_innersq) *
  //   (params[nparams].cut_outersq-params[nparams].cut_innersq) *
  //   (params[nparams].cut_outersq-params[nparams].cut_innersq);

  // flag type2param with either i,j = D,A or j,i = D,A
  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    for (int j = MAX(jlo,i); j <= jhi; j++) {
      // printf("nparams %d\n", nparams);
      type2param[i][j] = nparams; // do we need something here for donor/acceptor?
      printf("nparams %i\n", nparams);
      count++;
    }
  }
  nparams++;
  if (count == 0) {
      error->all(FLERR,"Incorrect args for pair coefficients");
    }
  // int count = 0;
  // for (int i = ilo; i <= ihi; i++)
  //   for (int j = MAX(jlo,i); j <= jhi; j++)
  //     for (int k = klo; k <= khi; k++) {
  //       if (donor_flag == 0) type2param[i][j][k] = nparams;
  //       else type2param[j][i][k] = nparams;
  //       count++;
  //     }
  // nparams++;

  // if (count == 0) error->all(FLERR,"Incorrect args for pair coefficients");
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void PairManybodyDonorAcceptorSFG::init_style()
{
  printf("\t\t\t\tinit style\n");
//   // molecular system required to use special list to find H atoms
//   // tags required to use special list
//   // pair newton on required since are looping over D atoms
//   //   and computing forces on A,H which may be on different procs

   if (atom->molecular == Atom::ATOMIC)
     error->all(FLERR,"Pair style hbond/dreiding requires molecular system");
   if (atom->tag_enable == 0)
     error->all(FLERR,"Pair style hbond/dreiding requires atom IDs");
   if (atom->map_style == Atom::MAP_NONE)
     error->all(FLERR,"Pair style hbond/dreiding requires an atom map, "
                "see atom_modify");
   if (force->newton_pair == 0)
     error->all(FLERR,"Pair style hbond/dreiding requires newton pair on");

//   // set donor[M]/acceptor[M] if any atom of type M is a donor/acceptor

//   int anyflag = 0;
//   int n = atom->ntypes;
//   for (int m = 1; m <= n; m++) donor[m] = acceptor[m] = 0;
//   for (int i = 1; i <= n; i++)
//     for (int j = 1; j <= n; j++)
//       for (int k = 1; k <= n; k++)
//         if (type2param[i][j][k] >= 0) {
//           anyflag = 1;
//           donor[i] = 1;
//           acceptor[j] = 1;
//         }

  // int anyflag = 0;
  // int n = atom->ntypes;
  // for (int m = 1; m <= n; m++) {
  //   donor[m] = acceptor[m] = 0;
  // }
  // for (int i = 1; i <= n; i++) {
  //   for (int j = 1; j <= n; j++) {
  //     if (type2param[i][j] >= 0) {
  //       anyflag = 1;
  //       donor[i] = 1;
  //       acceptor[j] = 1;
  //     }
  //   }
  // }

//   if (!anyflag) error->all(FLERR,"No pair hbond/dreiding coefficients set");

//   // set additional param values
//   // offset is for LJ only, angle term is not included

//   for (int m = 0; m < nparams; m++) {
//     params[m].lj1 = 60.0*params[m].epsilon*pow(params[m].sigma,12.0);
//     params[m].lj2 = 60.0*params[m].epsilon*pow(params[m].sigma,10.0);
//     params[m].lj3 = 5.0*params[m].epsilon*pow(params[m].sigma,12.0);
//     params[m].lj4 = 6.0*params[m].epsilon*pow(params[m].sigma,10.0);

//     /*
//     if (offset_flag) {
//       double ratio = params[m].sigma / params[m].cut_outer;
//       params[m].offset = params[m].epsilon *
//         ((2.0*pow(ratio,9.0)) - (3.0*pow(ratio,6.0)));
//     } else params[m].offset = 0.0;
//     */
//   }

//   // full neighbor list request

  neighbor->add_request(this, NeighConst::REQ_FULL);
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

// double PairManybodyDonorAcceptorSFG::init_one(int i, int j)
// {
//   printf("init one\n");
//   // int m;

//   // // return maximum cutoff for any K with I,J = D,A or J,I = D,A
//   // // donor/acceptor is not symmetric, IJ interaction != JI interaction

//   // double cut = 0.0;
//   // for (int k = 1; k <= atom->ntypes; k++) {
//   //   m = type2param[i][j][k];
//   //   if (m >= 0) cut = MAX(cut,params[m].cut_outer);
//   //   m = type2param[j][i][k];
//   //   if (m >= 0) cut = MAX(cut,params[m].cut_outer);
//   // }
//   // return cut;
// }

/* ---------------------------------------------------------------------- */

// double PairManybodyDonorAcceptorSFG::single(int i, int j, int itype, int jtype,
//                                    double rsq,
//                                    double /*factor_coul*/, double /*factor_lj*/,
//                                    double &fforce)
// {
//   int k,kk,ktype,knum,m;
//   tagint tagprev;
//   double eng,eng_lj,force_kernel,force_angle;
//   double rsq1,rsq2,r1,r2,c,s,ac,r2inv,r10inv,factor_hb;
//   double switch1,switch2;
//   double delr1[3],delr2[3];
//   tagint *klist;

//   double **x = atom->x;
//   int *type = atom->type;
//   double *special_lj = force->special_lj;

//   eng = 0.0;
//   fforce = 0;

//   // sanity check

//   if (!donor[itype]) return 0.0;
//   if (!acceptor[jtype]) return 0.0;

//   int molecular = atom->molecular;
//   if (molecular == Atom::MOLECULAR) {
//     klist = atom->special[i];
//     knum = atom->nspecial[i][0];
//   } else {
//     if (atom->molindex[i] < 0) return 0.0;
//     int imol = atom->molindex[i];
//     int iatom = atom->molatom[i];
//     Molecule **onemols = atom->avec->onemols;
//     klist = onemols[imol]->special[iatom];
//     knum = onemols[imol]->nspecial[iatom][0];
//     tagprev = atom->tag[i] - iatom - 1;
//   }

//   factor_hb = special_lj[sbmask(j)];

//   for (kk = 0; kk < knum; kk++) {
//     if (molecular == Atom::MOLECULAR) k = atom->map(klist[kk]);
//     else k = atom->map(klist[kk]+tagprev);

//     if (k < 0) continue;
//     ktype = type[k];
//     m = type2param[itype][jtype][ktype];
//     if (m < 0) continue;
//     const Param &pm = params[m];

//     delr1[0] = x[i][0] - x[k][0];
//     delr1[1] = x[i][1] - x[k][1];
//     delr1[2] = x[i][2] - x[k][2];
//     domain->minimum_image(delr1);
//     rsq1 = delr1[0]*delr1[0] + delr1[1]*delr1[1] + delr1[2]*delr1[2];
//     r1 = sqrt(rsq1);

//     delr2[0] = x[j][0] - x[k][0];
//     delr2[1] = x[j][1] - x[k][1];
//     delr2[2] = x[j][2] - x[k][2];
//     domain->minimum_image(delr2);
//     rsq2 = delr2[0]*delr2[0] + delr2[1]*delr2[1] + delr2[2]*delr2[2];
//     r2 = sqrt(rsq2);

//     // angle (cos and sin)

//     c = delr1[0]*delr2[0] + delr1[1]*delr2[1] + delr1[2]*delr2[2];
//     c /= r1*r2;
//     if (c > 1.0) c = 1.0;
//     if (c < -1.0) c = -1.0;
//     ac = acos(c);

//     if (ac < pm.cut_angle || ac > (2.0*MY_PI - pm.cut_angle)) return 0.0;
//     s = sqrt(1.0 - c*c);
//     if (s < SMALL) s = SMALL;

//     // LJ-specific kernel

//     r2inv = 1.0/rsq;
//     r10inv = r2inv*r2inv*r2inv*r2inv*r2inv;
//     force_kernel = r10inv*(pm.lj1*r2inv - pm.lj2)*r2inv * powint(c,pm.ap);
//     force_angle = pm.ap * r10inv*(pm.lj3*r2inv - pm.lj4) *
//       powint(c,pm.ap-1)*s;

//     // only lj part for now

//     eng_lj = r10inv*(pm.lj3*r2inv - pm.lj4);
//     if (rsq > pm.cut_innersq) {
//       switch1 = (pm.cut_outersq-rsq) * (pm.cut_outersq-rsq) *
//                 (pm.cut_outersq + 2.0*rsq - 3.0*pm.cut_innersq) / pm.denom_vdw;
//       switch2 = 12.0*rsq * (pm.cut_outersq-rsq) *
//                 (rsq-pm.cut_innersq) / pm.denom_vdw;
//       force_kernel = force_kernel*switch1 + eng_lj*switch2;
//       eng_lj *= switch1;
//     }

//     fforce += force_kernel*powint(c,pm.ap) + eng_lj*force_angle;
//     eng += eng_lj * powint(c,pm.ap) * factor_hb;
//   }

//   return eng;
// }
