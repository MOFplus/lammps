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

#include "pair_nb3b_test.h"

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

Pairnb3nTest::Pairnb3nTest(LAMMPS *lmp) : Pair(lmp)
{
  no_virial_fdotr_compute = 1;
  restartinfo = 0;
}

/* ---------------------------------------------------------------------- */

Pairnb3nTest::~Pairnb3nTest()
{
  if (allocated) {
    memory->destroy(setflag);
  }
}

/* ---------------------------------------------------------------------- */

void Pairnb3nTest::compute(int eflag, int vflag)
{
  int i,j,k,m,ii,jj,kk,inum,jnum,itype,jtype,ktype,iatom,imol;
  tagint tagprev;
  double fj[3],fk[3];
  int *ilist,*jlist,*numneigh,**firstneigh;

  double dx_ij, dy_ij, dz_ij, dx_ik, dy_ik, dz_ik;
  double xyz_ij[3], xyz_ik[3];
  double r_ij, r_ik, dot_a;
  double E, dot_a_ref, cut_ij, cut_ik, eps_a;
  double diff_cut_ij, diff_cut_ik, dE_dr_ij, dE_dr_ik, dE_da;
  double dr_ij_dj_x, dr_ij_dk_x, dr_ik_dj_x, dr_ik_dk_x;
  double partial_dot_jk, da_djx, da_dkx;
  double dot_ij, dot_ik, denominator1, denominator2;
  int mp, mpp;
  int id_mapper[5] = {0, 1, 2, 0, 1};

  E = 0.0;
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

  // ii -> center
  // jj -> edge
  // kk -> edge
  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    itype = type[i];
    if (center_typeid != itype) {
      continue;
    }
    jlist = firstneigh[i];
    jnum = numneigh[i];
    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      // factor_hb = special_lj[sbmask(j)];
      j &= NEIGHMASK;

      jtype = type[j];
      if (edge_typeid != jtype) {
        continue;
      }
      for (kk = jj; kk < jnum; kk++) {
        k = jlist[kk];
        k &= NEIGHMASK;

        ktype = type[k];
        if (edge_typeid != ktype) {
          continue;
        }
        
        // RIC calc
        xyz_ij[0] = dx_ij = x[j][0] - x[i][0];
        xyz_ij[1] = dy_ij = x[j][1] - x[i][1];
        xyz_ij[2] = dz_ij = x[j][2] - x[i][2];
        r_ij = sqrt(dx_ij*dx_ij + dy_ij*dy_ij + dz_ij*dz_ij); 

        xyz_ik[0] = dx_ik = x[k][0] - x[i][0];
        xyz_ik[1] = dy_ik = x[k][1] - x[i][1];
        xyz_ik[2] = dz_ik = x[k][2] - x[i][2];
        r_ik = sqrt(dx_ik*dx_ik + dy_ik*dy_ik + dz_ik*dz_ik);

        if (r_ij >= cutoff || r_ik >= cutoff) {
          continue;
        }
        dot_a = (dx_ij*dx_ik + dy_ij*dy_ik + dz_ij*dz_ik)/(r_ij * r_ik);
        
        // Energy calc
        dot_a_ref = cos(a_ref);
        cut_ij = (cos(M_PI/cutoff * r_ij)+1)/2;
        cut_ik = (cos(M_PI/cutoff * r_ik)+1)/2;
        eps_a = k_a * (dot_a - dot_a_ref)*(dot_a - dot_a_ref);
        E = eps_a  * cut_ij * cut_ik;
        
        // Force calc
        diff_cut_ij = -(M_PI/cutoff * sin((M_PI * r_ij)/cutoff))/2;
        diff_cut_ik = -(M_PI/cutoff * sin((M_PI * r_ik)/cutoff))/2;
        
        dE_dr_ij = eps_a * cut_ik * diff_cut_ij;
        dE_dr_ik = eps_a * cut_ij * diff_cut_ik;
        dE_da    = 2*k_a*(dot_a - dot_a_ref) * cut_ij * cut_ik;

        for (int m = 0; m < 3; m ++) {
          dot_ij = xyz_ij[0]*xyz_ij[0]+xyz_ij[1]*xyz_ij[1]+xyz_ij[2]*xyz_ij[2];
          dot_ik = xyz_ik[0]*xyz_ik[0]+xyz_ik[1]*xyz_ik[1]+xyz_ik[2]*xyz_ik[2];
          dr_ij_dj_x = xyz_ij[m]* 1/sqrt(dot_ij);
          dr_ij_dk_x = 0.0;
          dr_ik_dj_x = 0.0;
          dr_ik_dk_x = xyz_ik[m]* 1/sqrt(dot_ik);

          denominator1 = 1/(pow(dot_ij, 3/2) * sqrt(dot_ik));
          denominator2 = 1/(pow(dot_ik, 3/2) * sqrt(dot_ij));

          mp = id_mapper[m+1];
          mpp = id_mapper[m+2];
          
          partial_dot_jk = xyz_ij[mp]*xyz_ik[mp] + xyz_ij[mpp]*xyz_ik[mpp];
          da_djx = (xyz_ik[m]*(xyz_ij[mp]*xyz_ij[mp] + xyz_ij[mpp]*xyz_ij[mpp]) - xyz_ij[m]*partial_dot_jk) * denominator1;
          da_dkx = (xyz_ij[m]*(xyz_ik[mp]*xyz_ik[mp] + xyz_ik[mpp]*xyz_ik[mpp]) - xyz_ik[m]*partial_dot_jk) * denominator2;

          f[j][m] += -(dE_dr_ij*dr_ij_dj_x + dE_dr_ik*dr_ik_dj_x + dE_da*da_djx);
          f[k][m] += -(dE_dr_ij*dr_ij_dk_x + dE_dr_ik*dr_ik_dk_x + dE_da*da_dkx);
          f[i][m] -= (f[j][m] + f[k][m]);
        }

      fj[0] = f[j][0];
      fj[1] = f[j][1];
      fj[2] = f[j][2];

      fk[0] = f[k][0];
      fk[1] = f[k][1];
      fk[2] = f[k][2];
      double ecoul = 0.0; // why is ecoul AND evdW here but not in evtally4?
      if (evflag) ev_tally3(i, j, k, E, ecoul, fj, fk, xyz_ij, xyz_ik);
      }
    }
  }

}

/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

void Pairnb3nTest::allocate()
{
  allocated = 1;
  int n = atom->ntypes;

  // mark all setflag as set, since don't require pair_coeff of all I,J

  memory->create(setflag,n+1,n+1,"pair:setflag");
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++)
      setflag[i][j] = 1;

  memory->create(cutsq,n+1,n+1,"pair:cutsq"); //@KK does this belong here?
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void Pairnb3nTest::settings(int narg, char **arg)
{
  if (narg != 1) error->all(FLERR,"Wrong number of args given to nb3b, need 1!");
  cutoff = utils::numeric(FLERR,arg[0],false,lmp);
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void Pairnb3nTest::coeff(int narg, char **arg)
{
  // printf("%d\n",narg);
  if (narg != 5) {
    error->all(FLERR,"Incorrect args for pair coefficients, need 5 (2 ids, 1 center_id, 2 pars)");
  }
  if (!allocated) allocate();

  int ilo,ihi,jlo,jhi;
  utils::bounds(FLERR,arg[0],1,atom->ntypes,ilo,ihi,error); //@KK what do you do?
  utils::bounds(FLERR,arg[1],1,atom->ntypes,jlo,jhi,error); 

  int type_i, type_j;
  type_i = utils::numeric(FLERR,arg[0],false,lmp);
  type_j  = utils::numeric(FLERR,arg[1],false,lmp);

  int donor;
  if (strcmp(arg[2],"i") == 0) {
    donor = 0;
    center_typeid = MIN(type_i, type_j);
    edge_typeid = MAX(type_i, type_j);
  } else if (strcmp(arg[2],"j") == 0) {
    donor = 1;
    center_typeid = MAX(type_i, type_j);
    edge_typeid = MIN(type_i, type_j);
  } else {
    error->all(FLERR,"Incorrect args for pair coefficients, third flag must be either i or j");
  }

  k_a = utils::numeric(FLERR,arg[3],false,lmp); //@KK do I need "double k_a" what is the difference (impact of .h file)?
  a_ref = utils::numeric(FLERR,arg[4],false,lmp);
  
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void Pairnb3nTest::init_style()
{
//   // molecular system required to use special list to find H atoms
//   // tags required to use special list
//   // pair newton on required since are looping over D atoms
//   //   and computing forces on A,H which may be on different procs

   if (atom->molecular == Atom::ATOMIC)
     error->all(FLERR,"Pair style MOFFF/nb3b requires molecular system");
   if (atom->tag_enable == 0)
     error->all(FLERR,"Pair style MOFFF/nb3b requires atom IDs");
   if (atom->map_style == Atom::MAP_NONE)
     error->all(FLERR,"Pair style MOFFF/nb3b requires an atom map, "
                "see atom_modify");
   if (force->newton_pair == 0)
     error->all(FLERR,"Pair style MOFFF/nb3b requires newton pair on");

  neighbor->add_request(this, NeighConst::REQ_FULL);
}

