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

    RS RUB 2023
    part of pylmps

    Target: a fix that computes COMs from chunks and provides them as a 
    communicated array in python. In python we get forces fcom on these 
    COMs with some potential and provide forces to atoms

    derived from fix_spring_chunk and fix_python_invoke

------------------------------------------------------------------------- */

#include "fix_python_chunk.h"

#include "atom.h"
#include "comm.h"
#include "compute_chunk_atom.h"
#include "compute_com_chunk.h"
#include "error.h"
#include "lmppython.h"
#include "python_compat.h"
#include "python_utils.h"
#include "memory.h"
#include "modify.h"
// #include "respa.h"    // let us do RESPA at a later point 
#include "update.h"

#include <cmath>
#include <cstring>
#include <Python.h>   // IWYU pragma: export
#include <numpy/arrayobject.h>
#include <numpy/npy_common.h>

using namespace LAMMPS_NS;
using namespace FixConst;

#define SMALL 1.0e-10

/* ---------------------------------------------------------------------- */

FixPythonChunk::FixPythonChunk(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg),
  idchunk(nullptr), idcom(nullptr), fcom(nullptr)
{
  if (narg != 6) utils::missing_cmd_args(FLERR, "fix python/chunk", error);

  // restart_global = 1; // not sure but we leave out restart .. so just keep default ... which is??
  scalar_flag = 1;
  global_freq = 1;
  extscalar = 1;
  energy_global_flag = 1;
  // respa_level_support = 1;
  // ilevel_respa = 0;

  // RS  CHECK not shure if thermo_energy = 1 is needed or not
  //           it should make fix_modify energy yes possible to switch on
  //           this energy for minimization ... bit spring_chunk should also?
  //    needs to be tested

  // k_spring = utils::numeric(FLERR,arg[3],false,lmp); //remove later

  idchunk = utils::strdup(arg[3]);
  idcom = utils::strdup(arg[4]);

  epychunk = 0.0;
  nchunk = 0;

  // RS get Python function (the follwoing is from python/invoke)
  // ensure Python interpreter is initialized
  python->init();

  PyUtils::GIL lock;

  PyObject *pyMain = PyImport_AddModule("__main__");

  if (!pyMain) {
    PyUtils::Print_Errors();
    error->all(FLERR,"Could not initialize embedded Python");
  }

  char *fname = arg[5];
  pFunc = PyObject_GetAttrString(pyMain, fname);

  if (!pFunc) {
    PyUtils::Print_Errors();
    error->all(FLERR,"Could not find Python function");
  }

  lmpPtr = PY_VOID_POINTER(lmp);

}

/* ---------------------------------------------------------------------- */

FixPythonChunk::~FixPythonChunk()
{
  memory->destroy(fcom);

  // decrement lock counter in compute chunk/atom, it if still exists

  cchunk = dynamic_cast<ComputeChunkAtom *>(modify->get_compute_by_id(idchunk));
  if (cchunk) {
    cchunk->unlock(this);
    cchunk->lockcount--;
  }

  delete[] idchunk;
  delete[] idcom;

  PyUtils::GIL lock;
  Py_CLEAR(lmpPtr);
  Py_CLEAR(py_com);
  Py_CLEAR(py_fcom);
}

/* ---------------------------------------------------------------------- */

int FixPythonChunk::setmask()
{
  int mask = 0;
  mask |= POST_FORCE;
  // mask |= POST_FORCE_RESPA;
  mask |= MIN_POST_FORCE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixPythonChunk::init()
{
  // current indices for idchunk and idcom

  cchunk = dynamic_cast<ComputeChunkAtom *>(modify->get_compute_by_id(idchunk));
  if (!cchunk)
    error->all(FLERR,"Chunk/atom compute {} does not exist or is not chunk/atom style", idchunk);

  ccom = dynamic_cast<ComputeCOMChunk *>(modify->get_compute_by_id(idcom));
  if (!ccom)
    error->all(FLERR,"Com/chunk compute {} does not exist or is not com/chunk style", idcom);

  // check that idchunk is consistent with ccom->idchunk

  if (ccom && (strcmp(idchunk,ccom->idchunk) != 0))
    error->all(FLERR,"Fix python/chunk chunk ID {} not the same as compute com/chunk chunk ID {}",
               idchunk, ccom->idchunk);

  if (utils::strmatch(update->integrate_style,"^respa")) {
    error->all(FLERR, "RESPA currently not supported in fix python/chunk");
/*    ilevel_respa = (dynamic_cast<Respa *>(update->integrate))->nlevels-1;
    if (respa_level >= 0) ilevel_respa = MIN(respa_level,ilevel_respa); */
  }
}

/* ---------------------------------------------------------------------- */

void FixPythonChunk::setup(int vflag)
{
  if (utils::strmatch(update->integrate_style,"^verlet"))
    post_force(vflag);
  else {
    error->all(FLERR, "RESPA currently not supported in fix python/chunk");
  /*  (dynamic_cast<Respa *>(update->integrate))->copy_flevel_f(ilevel_respa);
    post_force_respa(vflag,ilevel_respa,0);
    (dynamic_cast<Respa *>(update->integrate))->copy_f_flevel(ilevel_respa); */
  }
}

/* ---------------------------------------------------------------------- */

void FixPythonChunk::min_setup(int vflag)
{
  post_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixPythonChunk::post_force(int /*vflag*/)
{
  int i,m;

  // check if first time cchunk will be queried via ccom
  // if so, lock idchunk for as long as this fix is in place
  // will be unlocked in destructor
  // necessary b/c this fix stores original COM

  if (fcom == nullptr) cchunk->lock(this,update->ntimestep,-1);

  // calculate current centers of mass for each chunk
  // extract pointers from idchunk and idcom
  
  ccom->compute_array();

  nchunk = cchunk->nchunk;
  int *ichunk = cchunk->ichunk;
  double *masstotal = ccom->masstotal;
  double **com = ccom->array;

  PyUtils::GIL lock;

  // generate fcom array if not exisiting
  // also make numpy arrays py_com and py_fcom for the data exchange with the python callback function
  if (fcom == nullptr) {
    memory->create(fcom,nchunk,3,"python/chunk:fcom");

    // this took me hours: this all stuff just segfaults if the import_array() is not called. 
    //    the next problem is that import_arry() does not compile --> bug in numpy
    //    see our mighty friend stackoverflow: https://stackoverflow.com/questions/52828873/how-does-import-array-in-numpy-c-api-work
    //    one of the answers was to use _import_array() instead ... tried it and it worked. not sure if that is a problem at some point

    if(PyArray_API == NULL) {
      _import_array(); 
    }

    printf("DEBUG DEBUG nchunk %d\n", nchunk);

    npy_intp dims[2]; 
    dims[0] = nchunk;
    dims[1] = 3;

    py_com = PyArray_SimpleNewFromData(2, dims, NPY_DOUBLE, (void *) com[0]);
    py_fcom = PyArray_SimpleNewFromData(2, dims, NPY_DOUBLE, (void *) fcom[0]); // this could be done only once
  }

  // call python callback to compute energy and force on coms, returns the fix energy
  PyObject * result = PyObject_CallFunction((PyObject*)pFunc, (char *) "OOO", (PyObject*)lmpPtr, py_com, py_fcom);

  if (!result) {
    PyUtils::Print_Errors();
    error->all(FLERR,"Fix python/chunk callback method failed");
  }

  epychunk = PyFloat_AsDouble(result);
  Py_CLEAR(result);

  /*
  // DEBUG DEBUG
  printf("DEBUG DEBUG this is fix python/chunk\n");
  for (i=0; i<10; i++) {
    printf ("  %3d %10.5f %10.5f %10.5f\n", i, fcom[i][0], fcom[i][1], fcom[i][2]);
  }
  printf ("DFEBUG DEBUG end");
  */

  // apply restoring force to atoms in each chunk

  double **f = atom->f;
  int *type = atom->type;
  double *mass = atom->mass;
  double *rmass = atom->rmass;
  int nlocal = atom->nlocal;

  double massone;

  if (rmass) {
    for (i = 0; i < nlocal; i++) {
      m = ichunk[i]-1;
      if (m < 0) continue;
      massone = rmass[i]/masstotal[m];
      f[i][0] += fcom[m][0]*massone;
      f[i][1] += fcom[m][1]*massone;
      f[i][2] += fcom[m][2]*massone;
    }
  } else {
    for (i = 0; i < nlocal; i++) {
      m = ichunk[i]-1;
      if (m < 0) continue;
      massone = mass[type[i]]/masstotal[m];
      f[i][0] += fcom[m][0]*massone;
      f[i][1] += fcom[m][1]*massone;
      f[i][2] += fcom[m][2]*massone;
    }
  }
}

/* ---------------------------------------------------------------------- */

void FixPythonChunk::post_force_respa(int vflag, int ilevel, int /*iloop*/)
{
  error->all(FLERR, "RESPA currently not supported in fix python/chunk");
  //  if (ilevel == ilevel_respa) post_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixPythonChunk::min_post_force(int vflag)
{
  post_force(vflag);
}

/* ----------------------------------------------------------------------
   writ number of chunks and position of original COM into restart
------------------------------------------------------------------------- */

/*
void FixSpringChunk::write_restart(FILE *fp)
{
  double n = nchunk;

  if (comm->me == 0) {
    int size = (3*n+1) * sizeof(double);
    fwrite(&size,sizeof(int),1,fp);
    fwrite(&n,sizeof(double),1,fp);
    fwrite(&com0[0][0],3*sizeof(double),nchunk,fp);
  }
}
*/

/* ----------------------------------------------------------------------
   use state info from restart file to restart the Fix
------------------------------------------------------------------------- */

/*
void FixSpringChunk::restart(char *buf)
{
  auto list = (double *) buf;
  int n = list[0];

  memory->destroy(com0);
  memory->destroy(fcom);

  cchunk = dynamic_cast<ComputeChunkAtom *>(modify->get_compute_by_id(idchunk));
  if (!cchunk)
    error->all(FLERR,"Chunk/atom compute {} does not exist or is not chunk/atom style", idchunk);

  nchunk = cchunk->setup_chunks();
  cchunk->compute_ichunk();
  memory->create(com0,nchunk,3,"spring/chunk:com0");
  memory->create(fcom,nchunk,3,"spring/chunk:fcom");

  if (n != nchunk) {
    if (comm->me == 0)
      error->warning(FLERR,"Number of chunks changed from {} to {}. Cannot use restart", n, nchunk);
    memory->destroy(com0);
    memory->destroy(fcom);
    nchunk = 1;
  } else {
    cchunk->lock(this,update->ntimestep,-1);
    memcpy(&com0[0][0],list+1,3*n*sizeof(double));
  }
}
*/

/* ----------------------------------------------------------------------
   energy of python chunks
------------------------------------------------------------------------- */

double FixPythonChunk::compute_scalar()
{
  return epychunk;
}
