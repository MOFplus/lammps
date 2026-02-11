/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government ret
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Babak Farhadi Jahromi (RUB)
------------------------------------------------------------------------- */

#include "compute_vacf_mw.h"

#include "atom.h"
#include "error.h"
#include "fix_store_state.h"
#include "group.h"
#include "modify.h"
#include "update.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

ComputeVACFMW::ComputeVACFMW(LAMMPS *lmp, int narg, char **arg) :
    Compute(lmp, narg, arg), id_fix(nullptr)
{
  if (narg < 4) error->all(FLERR, "Illegal compute vacf/mw command");

  scalar_flag = 1;
  extscalar = 0;
  create_attribute = 1;
  int nrezero = utils::inumeric(FLERR, arg[3], false, lmp);

  // create a new fix STORE style
  // id = compute-ID + COMPUTE_STORE, fix group = compute group

  id_fix = utils::strdup(id + std::string("_COMPUTE_STORE"));
  fix = dynamic_cast<FixStoreState *>(
      modify->add_fix(fmt::format("{} {} store/state {} vx vy vz", id_fix, group->names[igroup], nrezero)));

  // store current velocities in fix store array
  // skip if reset from restart file

  if (fix->restart_reset)
    fix->restart_reset = 0;
  else {
    double **voriginal = fix->array_atom;

    double **v = atom->v;
    int *mask = atom->mask;
    int nlocal = atom->nlocal;

    for (int i = 0; i < nlocal; i++)
      if (mask[i] & groupbit) {
        voriginal[i][0] = v[i][0];
        voriginal[i][1] = v[i][1];
        voriginal[i][2] = v[i][2];
      } else
        voriginal[i][0] = voriginal[i][1] = voriginal[i][2] = 0.0;
  }
}

/* ---------------------------------------------------------------------- */

ComputeVACFMW::~ComputeVACFMW()
{
  // check nfix in case all fixes have already been deleted

  if (modify->nfix) modify->delete_fix(id_fix);

  delete[] id_fix;
  delete[] vector;
}

/* ---------------------------------------------------------------------- */

void ComputeVACFMW::init()
{
  // set fix which stores original atom velocities

  fix = dynamic_cast<FixStoreState *>(modify->get_fix_by_id(id_fix));
  if (!fix) error->all(FLERR, "Could not find compute vacf/mw fix ID {}", id_fix);
}

/* ---------------------------------------------------------------------- */

double ComputeVACFMW::compute_scalar()
{
  invoked_scalar = update->ntimestep;

  double **voriginal = fix->array_atom;

  double **v = atom->v;
  double *mass = atom->mass;
  double *rmass = atom->rmass;
  int *mask = atom->mask;
  int *type = atom->type;
  int nlocal = atom->nlocal;

  double vxsq, vysq, vzsq, massone;
  double vacf;
  vacf = 0.0;

  for (int i = 0; i < nlocal; i++)
    if (mask[i] & groupbit) {
      if (rmass)
        massone = rmass[i];
      else
        massone = mass[type[i]];
      vxsq = v[i][0] * voriginal[i][0];
      vysq = v[i][1] * voriginal[i][1];
      vzsq = v[i][2] * voriginal[i][2];
      vacf += massone * (vxsq + vysq + vzsq);
    }

  MPI_Allreduce(&vacf, &scalar, 1, MPI_DOUBLE, MPI_SUM, world);
  return scalar;
}

/* ----------------------------------------------------------------------
   initialize one atom's storage values, called when atom is created
------------------------------------------------------------------------- */

void ComputeVACFMW::set_arrays(int i)
{
  double **voriginal = fix->array_atom;
  double **v = atom->v;
  voriginal[i][0] = v[i][0];
  voriginal[i][1] = v[i][1];
  voriginal[i][2] = v[i][2];
}
