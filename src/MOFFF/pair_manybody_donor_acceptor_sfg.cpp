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
   Contributing author: Kim Kirschenberger
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
  double f_jk, f_c1k, f_c2k;

  E_sfg = 0.0;
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

  // ii -> donors
  // jj -> acceptors
  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    itype = type[i];
    if (donor_typeid != itype) {
      continue;
    }
    jlist = firstneigh[i];
    jnum = numneigh[i];
    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      // factor_hb = special_lj[sbmask(j)];
      j &= NEIGHMASK;

      jtype = type[j];
      if (acceptor_typeid != jtype) {
        continue;
      }
      delx = x[i][0] - x[j][0];
      dely = x[i][1] - x[j][1];
      delz = x[i][2] - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz; 

      m = type2param[itype][jtype];
      if (m < 0) {
        continue;
      }
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
      if (sqrt(rsq) >= pm.cutoff_dsf) {
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

      alpha_dot = (r_DCC[0]*r_DA[0]+r_DCC[1]*r_DA[1]+r_DCC[2]*r_DA[2])/(norm_da * norm_dcc);
      beta_dot = (cc_cross[0]*r_DA[0]+cc_cross[1]*r_DA[1]+cc_cross[2]*r_DA[2])/(norm_da * norm_cc_cross);
      cut = pm.cutoff_dsf;

      // Energy calc
      exp_pi_r = exp(-a_p * (norm_da - rp_0)); 
      exp_pi_cut = exp(-a_p * (cut - rp_0));
      exp_sig_r = exp(-a_s * (norm_da - rs_0)); 
      exp_sig_cut = exp(-a_s * (cut - rs_0));
      D_pi_beta = beta_dot*beta_dot;
      D_sig_alpha = (1.5 + alpha_dot)/(2.5) * alpha_dot*alpha_dot;

      exp_pi_m1_r = exp_pi_r - 1;
      exp_pi_m1_cut = exp_pi_cut - 1;
      exp_sig_m1_r = exp_sig_r - 1;
      exp_sig_m1_cut = exp_sig_cut - 1;
      pre_pi = dp_0 * D_pi_beta;
      pre_sig = ds_0 * D_sig_alpha;

      E_r =  pre_sig * exp_sig_m1_r*exp_sig_m1_r + 
             pre_pi  * exp_pi_m1_r*exp_pi_m1_r;
      E_cut =  pre_sig * exp_sig_m1_cut*exp_sig_m1_cut + 
               pre_pi  * exp_pi_m1_cut*exp_pi_m1_cut;
      dE_dr_cut = -2 * (pre_sig * a_s * exp_sig_cut * exp_sig_m1_cut + pre_pi * a_p * exp_pi_cut * exp_pi_m1_cut);
      d2E_dr2_cut = 2 * (pre_sig * a_s*a_s * exp_sig_cut * (2*exp_sig_cut - 1) + 
                         pre_pi  * a_p*a_p * exp_pi_cut  * (2*exp_pi_cut  - 1));
      E_sfg = E_r - E_cut - (norm_da - cut)*dE_dr_cut - 0.5*(norm_da - cut)*(norm_da - cut)*d2E_dr2_cut;

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
        double f_jk = -(dE_dr * dr_dx + dE_dadot * da_dx + dE_dbdot * db_dx);
        f[j][k] += f_jk;
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
        double f_c1k = -(dE_dr * dr_dx + dE_dadot * da_dx + dE_dbdot * db_dx);
        f[c1_id][k] += f_c1k;

        // #c2
        if (k == 0) {
          db_dx = ((r_DA[1]*r_DC1[2] - r_DA[2]*r_DC1[1])*dbdx_term2 + (r_DC1[1]*cross_c02_20 + r_DC1[2]*(r_DC1[0]*r_DC2[2] - r_DC1[2]*r_DC2[0]))*dbdx_term1)/dbdx_termc;
        } else if (k == 1) {
          db_dx = ((-r_DA[0]*r_DC1[2] + r_DA[2]*r_DC1[0])*dbdx_term2 - (r_DC1[0]*cross_c02_20 - r_DC1[2]*cross_c11_c21)*dbdx_term1)/dbdx_termc;
        } else if (k == 2) {
          db_dx = ((r_DA[0]*r_DC1[1] - r_DA[1]*r_DC1[0])*dbdx_term2 - (r_DC1[0]*(r_DC1[0]*r_DC2[2] - r_DC1[2]*r_DC2[0]) + r_DC1[1]*cross_c11_c21)*dbdx_term1)/dbdx_termc;
        }
        double f_c2k = -(dE_dr * dr_dx + dE_dadot * da_dx + dE_dbdot * db_dx);
        f[c2_id][k] += f_c2k;
    
        // #donor
        f[i][k] += - (f_jk + f_c1k + f_c2k);
      }

    fj[0] = f[j][0];
    fj[1] = f[j][1];
    fj[2] = f[j][2];

    fc1[0] = f[c1_id][0];
    fc1[1] = f[c1_id][1];
    fc1[2] = f[c1_id][2];

    fc2[0] = f[c2_id][0];
    fc2[1] = f[c2_id][1];
    fc2[2] = f[c2_id][2];
    if (evflag) ev_tally4(j,c1_id,c2_id,i,E_sfg,fj,fc1,fc2,r_DA,r_DC1,r_DC2);
    }
  }

}

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

  int ilo,ihi,jlo,jhi;
  utils::bounds(FLERR,arg[0],1,atom->ntypes,ilo,ihi,error); 
  utils::bounds(FLERR,arg[1],1,atom->ntypes,jlo,jhi,error); 

  int type_i, type_j;
  type_i = utils::numeric(FLERR,arg[0],false,lmp);
  type_j  = utils::numeric(FLERR,arg[1],false,lmp);

  int donor;
  if (strcmp(arg[2],"i") == 0) {
    donor = 0;
    donor_typeid = MIN(type_i, type_j);
    acceptor_typeid = MAX(type_i, type_j);
  } else if (strcmp(arg[2],"j") == 0) {
    donor = 1;
    donor_typeid = MAX(type_i, type_j);
    acceptor_typeid = MIN(type_i, type_j);
  } else {
    error->all(FLERR,"Incorrect args for pair coefficients, third flag must be either i or j");
  }

  double dp_0 = utils::numeric(FLERR,arg[3],false,lmp);
  double ds_0 = utils::numeric(FLERR,arg[4],false,lmp);
  double a_p  = utils::numeric(FLERR,arg[5],false,lmp);
  double a_s  = utils::numeric(FLERR,arg[6],false,lmp);
  double rp_0 = utils::numeric(FLERR,arg[7],false,lmp);
  double rs_0 = utils::numeric(FLERR,arg[8],false,lmp);

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

  // flag type2param with either i,j = D,A or j,i = D,A
  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    for (int j = MAX(jlo,i); j <= jhi; j++) {
      if (donor == 0){
        type2param[i][j] = nparams; 
      } else if (donor == 1) {
        type2param[j][i] = nparams;
      }
      count++;
    }
  }
  nparams++;
  if (count == 0) {
      error->all(FLERR,"Incorrect args for pair coefficients");
    }
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void PairManybodyDonorAcceptorSFG::init_style()
{
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

  neighbor->add_request(this, NeighConst::REQ_FULL);
}

