// clang-format off
/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.org, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Pierre de Buyl (KU Leuven)
                        Rochus Schmid (RUB)
                        Note: this is a rip off of Pierre de Buyl's h5md dump
                              to write hdf5 based mfp5 files. Thanks to Pierre for the clear code!
------------------------------------------------------------------------- */

/* 
RS Notes 2022 
Note that this works only together with pylmps since the hdf5 must be generated somehow already.
We only open the file and the datasets and write the data out.

There are two major hacks in this code:
- we grab the data from the fix reaxff/bonds (usually this writes an ascii file .. now it also writes an array and this dump pushes the data to the hdf5)
- we write data from the thermo log output also to the dump

this implies changes in REAXFF/fix_reaxff_bonds.cpp and in thermo.cpp
WARNING: i am pretty sure that there are better (proper) ways to do this and the current construction is only becasue i am too stupid to understand how to do it right
         BUT it works 

*/

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>
#include "hdf5.h"
#include "dump_mfp5.h"
#include "domain.h"
#include "atom.h"
#include "update.h"
#include "group.h"
#include "output.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "version.h"
#include "thermo.h"

#include "fix_reaxff_bonds.h"
#include "modify.h"

using namespace LAMMPS_NS;

#define MYMIN(a,b) ((a) < (b) ? (a) : (b))
#define MYMAX(a,b) ((a) > (b) ? (a) : (b))

/** Scan common options for the dump elements
 */
static int element_args(int narg, char **arg, int *every)
{
  int iarg=0;
  while (iarg<narg) {
    if (strcmp(arg[iarg], "every")==0) {
      if (narg<2) return -1;
      *every = atoi(arg[iarg+1]);
      iarg+=2;
    } else {
      break;
    }
  }
  return iarg;
}

/* ---------------------------------------------------------------------- */

DumpMFP5::DumpMFP5(LAMMPS *lmp, int narg, char **arg) : Dump(lmp, narg, arg)
{
  if (narg<6) error->all(FLERR,"Illegal dump mfp5 command");
  if (binary || compressed || multifile || multiproc)
    error->all(FLERR,"Invalid dump mfp5 filename");

//  if (domain->triclinic!=0)
//    error->all(FLERR,"Invalid domain for dump mfp5. Only orthorombic domains supported.");

  sort_flag = 1;
  sortcol = 0;
  format_default = nullptr;
  flush_flag = 0;
  unwrap_flag = 0;
  stage_name= nullptr;

  every_dump = utils::inumeric(FLERR,arg[3],false,lmp);
  
  every_xyz = -1;
  every_image = -1;
  every_vel = -1;
  every_forces = -1;
  every_charges = -1;
  every_cell = -1;
  every_restart = -1;
  every_thermo = -1;
  every_bond = -1;

  dump_count = 0; // this counter is only to syncronize the thermo dumping becasue thermo is done after dump in the cycle

  int iarg=5;
  int n_parsed, default_every;
  if (every_dump==0) default_every=0; else default_every=1;

  size_one = 0;
  while (iarg<narg) {
    if (strcmp(arg[iarg], "xyz")==0) {
      every_xyz=default_every;
      iarg+=1;
      n_parsed = element_args(narg-iarg, &arg[iarg], &every_xyz);
      if (n_parsed<0) error->all(FLERR, "Illegal dump mfp5 command");
      iarg += n_parsed;
      size_one+=domain->dimension;
    } else if (strcmp(arg[iarg], "stage")==0) {
      if (iarg+1>=narg) {
        error->all(FLERR, "Invalid number of arguments in dump mfp5");
      }
      if (stage_name==nullptr) {
        stage_name = new char[strlen(arg[iarg+1])+1]; // RS 7.5.25 this was a bug I think ... not sure if the second +1 is needed
        strcpy(stage_name, arg[iarg+1]);
      } else {
        error->all(FLERR, "Illegal dump mfp5 command: stage name argument repeated");
      }
      iarg+=2;
    } else if (strcmp(arg[iarg], "img")==0) {
      if (every_xyz<0) error->all(FLERR, "Illegal dump mfp5 command");
      iarg+=1;
      every_image = every_xyz;
      size_one+=domain->dimension;
    } else if (strcmp(arg[iarg], "vel")==0) {
      every_vel = default_every;
      iarg+=1;
      n_parsed = element_args(narg-iarg, &arg[iarg], &every_vel);
      if (n_parsed<0) error->all(FLERR, "Illegal dump mfp5 command");
      iarg += n_parsed;
      size_one+=domain->dimension;
    } else if (strcmp(arg[iarg], "forces")==0) {
      every_forces = default_every;
      iarg+=1;
      n_parsed = element_args(narg-iarg, &arg[iarg], &every_forces);
      if (n_parsed<0) error->all(FLERR, "Illegal dump mfp5 command");
      iarg += n_parsed;
      size_one+=domain->dimension;
    } else if (strcmp(arg[iarg], "charges")==0) {
      if (!atom->q_flag)
        error->all(FLERR, "Requesting non-allocated quantity q in dump_mfp5");
      every_charges = default_every;
      iarg+=1;
      n_parsed = element_args(narg-iarg, &arg[iarg], &every_charges);
      if (n_parsed<0) error->all(FLERR, "Illegal dump mfp5 command");
      iarg += n_parsed;
      size_one+=1;
    } else if (strcmp(arg[iarg], "cell")==0) {
      every_cell = default_every;
      iarg+=1;
      n_parsed = element_args(narg-iarg, &arg[iarg], &every_cell);
      if (n_parsed<0) error->all(FLERR, "Illegal dump mfp5 command");
      iarg += n_parsed;
    } else if (strcmp(arg[iarg], "restart")==0) {
      every_restart = default_every;
      iarg+=1;
      n_parsed = element_args(narg-iarg, &arg[iarg], &every_restart);
      if (n_parsed<0) error->all(FLERR, "Illegal dump mfp5 command");
      iarg += n_parsed;
    } else if (strcmp(arg[iarg], "thermo")==0) {
      every_thermo = default_every;
      iarg+=1;
      n_parsed = element_args(narg-iarg, &arg[iarg], &every_thermo);
      if (n_parsed<0) error->all(FLERR, "Illegal dump mfp5 command");
    } else if (strcmp(arg[iarg], "bond")==0) {
      every_bond = default_every;
      iarg+=1;
      n_parsed = element_args(narg-iarg, &arg[iarg], &every_bond);
      if (n_parsed<0) error->all(FLERR, "Illegal dump mfp5 command");
      iarg += n_parsed;
    } else {
      printf("DEBUG iarg: %d arg[iarg] %s\n", iarg, arg[iarg]);
      error->all(FLERR, "Invalid argument to dump mfp5");
    }
  }

  // fix up size_one becasue we need space for xyz/val in case only restart is chosen
  if (every_restart > 0) {
    if (every_xyz <0) size_one+=domain->dimension;
    if (every_vel <0) size_one+=domain->dimension;
    if (every_image <0) size_one+=domain->dimension; // RS add image to restart info (we mgiht switch this off if nonperiodic)
  }

  // if bond are written get corresponding fix
  // TBI check if fix exists .. we should take care for this with pylmps but a safety check would be good
  // we do this here depending on whether every_bond >= 0 .. this should be set only in case of reaxff 
  // which implies that the fix is exisiting
  if (every_bond>=0) {
    printf("DEBUG DEBUG .. using data from fix reaxff/bond\n");
    rxbfix = (FixReaxFFBonds *) modify->get_fix_by_style("reaxff/bonds")[0]; // pick first element .. there should only be one such fix
    printf("DEBUG DEBUG size of nmaxbonds %d\n", rxbfix->nbondmax);
  }

//  printf ("size_one is %d\n", size_one);

//  printf("every_xyz %d\n", every_xyz);
//  printf("every_image %d\n", every_image);
//  printf("every_vel %d\n", every_vel);
//  printf("every_cell %d\n", every_cell);
//  printf("every_forces %d\n", every_forces);
//  printf("every_charges %d\n", every_charges);
//  printf("every_restart %d\n", every_restart);
//  printf("every_thermo %d\n", every_thermo);


  // allocate global array for atom coords

  bigint n = group->count(igroup);
  natoms = static_cast<int> (n);

  if ((every_xyz>=0) || (every_restart>=0)) {
    memory->create(dump_xyz,domain->dimension*natoms,"dump:xyz");
    //printf ("dump:xyz allocated\n");
  }
  if ((every_image>=0) || (every_restart>=0)) {
    memory->create(dump_img,domain->dimension*natoms,"dump:img");
    //printf ("dump:xyz_img allocated\n");
  }
  if ((every_vel>=0) || (every_restart>=0)) {
    memory->create(dump_vel,domain->dimension*natoms,"dump:vel");
    //printf ("dump:vel allocated\n");
  }
  if (every_forces>=0) {
    memory->create(dump_forces,domain->dimension*natoms,"dump:forces");
    //printf ("dump:forces allocated\n");
  }
  if (every_charges>=0) {
    memory->create(dump_charges,natoms,"dump:charges");
    //printf ("dump:charges allocated\n");
  }
  

  // RS here the file is opened .. we need to see if we can just pass the hid_t of the hdf5 file and access it
  openfile();
  ntotal = 0;
}

/* ---------------------------------------------------------------------- */

DumpMFP5::~DumpMFP5()
{

  int statcode;

  if (every_xyz>=0 || every_restart>= 0) 
    memory->destroy(dump_xyz);
  if (every_xyz>=0) {
    if (me==0) H5Dclose(xyz_dset);    
  }
  if (every_image>=0 || every_restart>= 0) 
    memory->destroy(dump_img);
  if (every_image >=0) {
    if (me==0) H5Dclose(img_dset);    
  }
  if (every_vel>=0 || every_restart>=0) 
    memory->destroy(dump_vel);
  if (every_vel>=0) { 
    if (me==0) H5Dclose(vel_dset);    
  }
  if (every_forces>=0) {
    memory->destroy(dump_forces);
    if (me==0) H5Dclose(forces_dset);    
  }
  if (every_charges>=0) {
    memory->destroy(dump_charges);
    if (me==0) H5Dclose(charges_dset);    
  }
  if (every_cell>=0) {
    if (me==0) H5Dclose(cell_dset);    
  }
  if (every_thermo>=0) {
    if (me==0) {
      // write one more frame of thermo data to wrap up (thermo is called after dump so the last step is missing)
      statcode = append_data(thermo_dset, 2, output->thermo->thermo_values);
      //printf("after append thermo : %d\n" , statcode);
      H5Dclose(thermo_dset);
    }    
  }
  if (every_bond>=0) {
    if (me==0) {
      H5Dclose(bondtab_dset);
      H5Dclose(bondord_dset);
    }    
  }  if (every_restart>=0 && me==0) {
    H5Dclose(rest_xyz_dset);    
    H5Dclose(rest_vel_dset);    
    H5Dclose(rest_cell_dset);
    H5Dclose(rest_img_dset);    
  }

  if (me==0){
    H5Gclose(traj_group);
    H5Gclose(stage_group);
    H5Gclose(restart_group);
    H5Fclose(mfp5file);
    //printf("mfp5 dump file closed\n");
  }
}

/* ---------------------------------------------------------------------- */

void DumpMFP5::init_style()
{
  if (sort_flag == 0 || sortcol != 0)
    error->all(FLERR,"Dump mfp5 requires sorting by atom ID");
}

/* ---------------------------------------------------------------------- */

void DumpMFP5::openfile()
{
  int dims[2];

  // DEBUG
  int i;
  ssize_t len;
  hid_t root_group;
  herr_t err;
  hsize_t nobj;
  char memb_name[256];

  if (me == 0) {
    // me == 0 _> do only on master node
    
    mfp5file = H5Fopen(filename, H5F_ACC_RDWR, H5P_DEFAULT);
    //printf("file %s opened \n", filename);

    /* // DEBUG DBEUG
    root_group = H5Gopen(mfp5file, "/", H5P_DEFAULT);
    err = H5Gget_num_objs(root_group, &nobj);
    for (i = 0; i < nobj; i++) {
      len = H5Gget_objname_by_idx(root_group, (hsize_t)i, memb_name, (size_t)256 );
      printf("objname : %s\n", memb_name);
    }
    H5Gclose(root_group);
    // DEBUG DEBUG */

    stage_group = H5Gopen(mfp5file, stage_name, H5P_DEFAULT);
    //printf("group %s opened\n", stage_name);
    traj_group  = H5Gopen(stage_group, "traj", H5P_DEFAULT);
    restart_group = H5Gopen(stage_group, "restart", H5P_DEFAULT);
    //printf("mfp5 file opened   %d %d %d %d\n", mfp5file, stage_group, traj_group, restart_group);

    if (every_xyz>0) {
      xyz_dset    = H5Dopen(traj_group, "xyz", H5P_DEFAULT);
      //printf("mfp5 xyz dset opened\n");
    }
    if (every_image>0) {
      img_dset    = H5Dopen(traj_group, "img", H5P_DEFAULT);
      //printf("mfp5 img dset opened\n");
    }
    if (every_vel>0) {
      vel_dset    = H5Dopen(traj_group, "vel", H5P_DEFAULT);
      //printf("mfp5 vel dset opened\n");
    }
    if (every_forces>0) {
      forces_dset = H5Dopen(traj_group, "forces", H5P_DEFAULT);
      //printf("mfp5 forces dset opened\n");
    }
    if (every_charges>0) {
      charges_dset = H5Dopen(traj_group, "charges", H5P_DEFAULT);
      //printf("mfp5 charges dset opened\n");
    }
    if (every_cell>0) {
      cell_dset = H5Dopen(traj_group, "cell", H5P_DEFAULT);
      //printf("mfp5 cell dset opened\n");
    }
    if (every_thermo>0) {
      thermo_dset = H5Dopen(traj_group, "thermo", H5P_DEFAULT);
      //printf("mfp5 thermo dset opened\n");
    }
    if (every_bond>0) {
      bondtab_dset = H5Dopen(traj_group, "bondtab", H5P_DEFAULT);
      bondord_dset = H5Dopen(traj_group, "bondord", H5P_DEFAULT);
      //printf("mfp5 bond dset opened\n");
    }
    if (every_restart>0) {
      rest_xyz_dset = H5Dopen(restart_group, "xyz", H5P_DEFAULT);
      rest_vel_dset = H5Dopen(restart_group, "vel", H5P_DEFAULT);
      rest_cell_dset = H5Dopen(restart_group, "cell", H5P_DEFAULT);
      rest_img_dset = H5Dopen(restart_group, "img", H5P_DEFAULT);
      //printf("mfp5 restart dsets opened\n");
    }

    dims[0] = natoms;
    dims[1] = domain->dimension;
  }
}

/* ---------------------------------------------------------------------- 

void DumpMFP5::write_header(bigint nbig)
{
  return;
}

 ---------------------------------------------------------------------- */

void DumpMFP5::pack(tagint *ids)
{
  int m,n;

  tagint *tag = atom->tag;
  double **x = atom->x;
  double **v = atom->v;
  double **f = atom->f;
  double *q = atom->q;

  imageint *image = atom->image;

  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  int dim=domain->dimension;

  double xprd = domain->xprd;
  double yprd = domain->yprd;
  double zprd = domain->zprd;

  m = n = 0;
  for (int i = 0; i < nlocal; i++)
    if (mask[i] & groupbit) {
      if (every_xyz>=0 || every_restart>=0) {
        int ix = (image[i] & IMGMASK) - IMGMAX;
        int iy = (image[i] >> IMGBITS & IMGMASK) - IMGMAX;
        int iz = (image[i] >> IMG2BITS) - IMGMAX;
        if (unwrap_flag == 1) {
          buf[m++] = (x[i][0] + ix * xprd);
          buf[m++] = (x[i][1] + iy * yprd);
          if (dim>2) buf[m++] = (x[i][2] + iz * zprd);
        } else {
          buf[m++] = x[i][0];
          buf[m++] = x[i][1];
          if (dim>2) buf[m++] = x[i][2];
        }
        if (every_image>=0 || every_restart>=0) {
          buf[m++] = ix;
          buf[m++] = iy;
          if (dim>2) buf[m++] = iz;
        }
      }
      if (every_vel>=0 || every_restart>=0) {
        buf[m++] = v[i][0];
        buf[m++] = v[i][1];
        if (dim>2) buf[m++] = v[i][2];
      }
      if (every_forces>=0) {
        buf[m++] = f[i][0];
        buf[m++] = f[i][1];
        if (dim>2) buf[m++] = f[i][2];
      }
      if (every_charges>=0) {
        buf[m++] = q[i];
      }

      ids[n++] = tag[i];
    }
}

/* ---------------------------------------------------------------------- */

void DumpMFP5::write_data(int n, double *mybuf)
{
  // copy buf atom coords into global array

  int m = 0;
  int dim = domain->dimension;
  int k = dim*ntotal;
  int k_img = dim*ntotal;
  int k_vel = dim*ntotal;
  int k_frc = dim*ntotal;
  int k_chg = ntotal;

  for (int i = 0; i < n; i++) {
    if (every_xyz>=0 || every_restart>=0) {
      for (int j=0; j<dim; j++) {
        dump_xyz[k++] = mybuf[m++];
      }
      if (every_image>=0 || every_restart>=0)
        for (int j=0; j<dim; j++) {
          dump_img[k_img++] = mybuf[m++];
        }
    }
    if (every_vel>=0 || every_restart>=0)
      for (int j=0; j<dim; j++) {
        dump_vel[k_vel++] = mybuf[m++];
      }
    if (every_forces>=0)
      for (int j=0; j<dim; j++) {
        dump_forces[k_frc++] = mybuf[m++];
      }
    if (every_charges>=0)
      dump_charges[k_chg++] = mybuf[m++];
    ntotal++;
  }

  // if last chunk of atoms in this snapshot, write global arrays to file

  if (ntotal == natoms) {
    if (every_xyz>0 || every_restart>0) {
      write_frame();
      ntotal = 0;
    } 
    /*else {
      write_fixed_frame();
    }
    */
  }
}

/* ---------------------------------------------------------------------- */

int DumpMFP5::modify_param(int narg, char **arg)
{
  if (strcmp(arg[0],"unwrap") == 0) {
    if (narg < 2) error->all(FLERR,"Illegal dump_modify command");
    if (strcmp(arg[1],"yes") == 0) unwrap_flag = 1;
    else if (strcmp(arg[1],"no") == 0) unwrap_flag = 0;
    else error->all(FLERR,"Illegal dump_modify command");
    return 2;
  }
  return 0;
}

/* ---------------------------------------------------------------------- */

void DumpMFP5::write_frame()
{
  int local_step;
  double local_time;
  double cell[9];
  int i;
  int statcode;

  local_step = update->ntimestep;
  local_time = local_step * update->dt;
  for (i=0; i<9; i++) cell[i] = 0.0;
  if (!domain->triclinic) {
    //RS we follow here the computes in thermo 
    // orthorombic box
    cell[0] = domain->xprd;
    cell[4] = domain->yprd;
    cell[8] = domain->zprd;
  } else {
    // triclinic box: use domain->h[6] which is cell in Voigt notation
    // REMARK: in molsys the convention is to have the three vactors a,b,c as rows(!) of the cell matrix
    //         so we have to transpose
    // diagonal
    cell[0] = domain->h[0];
    cell[4] = domain->h[1];
    cell[8] = domain->h[2];
    // off-diagonal
    cell[3] = domain->h[3];
    cell[6] = domain->h[4];
    cell[7] = domain->h[5];    
  }
  
  if (every_xyz>0) {
    if (local_step % (every_xyz*every_dump) == 0) {
      if (dump_count == 0) {
       statcode = write_data(xyz_dset, 3, dump_xyz); 
      }
      else {
       statcode = append_data(xyz_dset, 3, dump_xyz); 
      }
    }
  }
  if (every_image>0) {
    if (local_step % (every_image*every_dump) == 0) {
      if (dump_count == 0) {
       statcode = write_data_int(img_dset, 3, dump_img); 
      }
      else {
       statcode = append_data_int(img_dset, 3, dump_img); 
      }
    }
  }
  if (every_cell>0 && local_step % (every_cell*every_dump) == 0) {
    if (dump_count == 0) {
      statcode = write_data(cell_dset, 3, cell); 
    }
    else {
      statcode = append_data(cell_dset, 3, cell); 
    }
  }  
  if (every_vel>0 && local_step % (every_vel*every_dump) == 0) {
    if (dump_count == 0) {
      statcode = write_data(vel_dset, 3, dump_vel); 
    }
    else {
      statcode = append_data(vel_dset, 3, dump_vel); 
    }
  }
  if (every_forces>0 && local_step % (every_forces*every_dump) == 0) {
    if (dump_count == 0) {
      statcode = write_data(forces_dset, 3, dump_forces); 
    }
    else {
      statcode = append_data(forces_dset, 3, dump_forces); 
    }
  }
  if (every_charges>0 && local_step % (every_charges*every_dump) == 0) {
    if (dump_count == 0) {
      statcode = write_data(charges_dset, 2, dump_charges); 
    }
    else {
      statcode = append_data(charges_dset, 2, dump_charges); 
    }
  }
  if (every_thermo>0 && local_step % (every_thermo*every_dump) == 0) {
    // take array of thermo_values from thermo object (public array)
    if (dump_count == 1) {
      statcode = write_data(thermo_dset, 2, output->thermo->thermo_values);
    } else if (dump_count > 1) {
      statcode = append_data(thermo_dset, 2, output->thermo->thermo_values);
    }
  }
  if (every_bond>0 && local_step % (every_bond*every_dump) == 0) {
    // take array of thermo_values from thermo object (public array)
    if (dump_count == 0) {
      statcode = write_data_int(bondtab_dset, 3, rxbfix->bondtab);
      statcode = write_data(bondord_dset, 2, rxbfix->bondord);
    } else {
      statcode = append_data_int(bondtab_dset, 3, rxbfix->bondtab);
      statcode = append_data(bondord_dset, 2, rxbfix->bondord);
    }
  }


  if (every_restart>0){
    if (local_step % (every_restart*every_dump) == 0) {
      statcode = write_data(rest_xyz_dset, 2, dump_xyz);
      statcode = write_data(rest_vel_dset, 2, dump_vel);
      statcode = write_data(rest_cell_dset, 2, cell);
      statcode = write_data_int(rest_img_dset, 2, dump_img);
    }
  }

  // increment counter
  dump_count += 1;

  H5Fflush(mfp5file, H5F_SCOPE_GLOBAL);
  //printf ("This is dump_mfp5 in timestp %d\n", local_step);
}

int DumpMFP5::append_data(hid_t dset, int rank, double *dump)
{
  herr_t  status;
  hsize_t dims[rank], start[rank], count[rank];
  hid_t   fspace, mspace;
  int i;
  
  fspace = H5Dget_space(dset);
  // get current dims
  H5Sget_simple_extent_dims(fspace, dims, nullptr);
  // increment by one frame
  dims[0] += 1;
  status = H5Dset_extent(dset, dims);
  H5Sclose(fspace);
  if (status<0){
    printf("Extending mfp5 dataset went wrong! status is %d\n", status);
    return -1;
  }
  // Now get fspace again
  fspace = H5Dget_space(dset);
  // create start and offset
  start[0] = dims[0]-1;
  count[0] = 1;
  for (i=1; i<rank; i++) {
    start[i] = 0;
    count[i] = dims[i];
  }
  // select part of file to be writen
  status = H5Sselect_hyperslab(fspace, H5S_SELECT_SET, start, nullptr, count, nullptr);
  if (status<0){
    printf("Selecting hyperslab went wrong! status is %d\n", status);
    H5Sclose(fspace);
    return -2;
  }
  // generate a mspace for the data in memory
  mspace = H5Screate_simple(rank-1, dims+1, nullptr);
  // write the data
  status = H5Dwrite(dset, H5T_IEEE_F64LE, mspace, fspace, H5P_DEFAULT, dump);
  // close selections
  H5Sclose(fspace);
  H5Sclose(mspace);
  if (status<0){
    printf("Writing data went wrong! status is %d\n", status);
    return -3;
  }      
return 0;
}

int DumpMFP5::write_data(hid_t dset, int rank, double *dump)
{
  herr_t  status;
  
  // write the data
  status = H5Dwrite(dset, H5T_IEEE_F64LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dump);
  if (status<0){
    printf("Writing data went wrong! status is %d\n", status);
    return -3;
  }      
return 0;
}

int DumpMFP5::append_data_int(hid_t dset, int rank, int *dump)
{
  herr_t  status;
  hsize_t dims[rank], start[rank], count[rank];
  hid_t   fspace, mspace;
  int i;
  
  fspace = H5Dget_space(dset);
  // get current dims
  H5Sget_simple_extent_dims(fspace, dims, nullptr);
  // increment by one frame
  dims[0] += 1;
  status = H5Dset_extent(dset, dims);
  H5Sclose(fspace);
  if (status<0){
    printf("Extending mfp5 dataset went wrong! status is %d\n", status);
    return -1;
  }
  // Now get fspace again
  fspace = H5Dget_space(dset);
  // create start and offset
  start[0] = dims[0]-1;
  count[0] = 1;
  for (i=1; i<rank; i++) {
    start[i] = 0;
    count[i] = dims[i];
  }
  // select part of file to be writen
  status = H5Sselect_hyperslab(fspace, H5S_SELECT_SET, start, nullptr, count, nullptr);
  if (status<0){
    printf("Selecting hyperslab went wrong! status is %d\n", status);
    H5Sclose(fspace);
    return -2;
  }
  // generate a mspace for the data in memory
  mspace = H5Screate_simple(rank-1, dims+1, nullptr);
  // write the data
  status = H5Dwrite(dset, H5T_STD_I32LE, mspace, fspace, H5P_DEFAULT, dump);
  // close selections
  H5Sclose(fspace);
  H5Sclose(mspace);
  if (status<0){
    printf("Writing data went wrong! status is %d\n", status);
    return -3;
  }      
return 0;
}

int DumpMFP5::write_data_int(hid_t dset, int rank, int *dump)
{
  herr_t  status;
  
  // write the data
  status = H5Dwrite(dset, H5T_STD_I32LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dump);
  if (status<0){
    printf("Writing data went wrong! status is %d\n", status);
    return -3;
  }      
return 0;
}
