#if !defined(__PETSCIMPL_H)
#define	__PETSCIMPL_H

#include "petscpc.h"

#if PETSC_VERSION_MINOR==6
#include "permon/private/petsc/3.6/mat/aij.h"
#include "permon/private/petsc/3.6/mat/dense.h"
#include "permon/private/petsc/3.6/mat/mpiaij.h"
#include "permon/private/petsc/3.6/mat/matnestimpl.h"
#include "permon/private/petsc/3.6/mat/normm.h"
#include "permon/private/petsc/3.6/mat/sbaij.h"
#include "permon/private/petsc/3.6/mat/shell.h"
#include "permon/private/petsc/3.6/mat/transm.h"
#include "permon/private/petsc/3.6/pc/redundant.h"
#elif PETSC_VERSION_MINOR==7
#include "permon/private/petsc/3.7/mat/aij.h"
#include "permon/private/petsc/3.7/mat/dense.h"
#include "permon/private/petsc/3.7/mat/mpiaij.h"
#include "permon/private/petsc/3.7/mat/matnestimpl.h"
#include "permon/private/petsc/3.6/mat/normm.h"
#include "permon/private/petsc/3.7/mat/sbaij.h"
#include "permon/private/petsc/3.7/mat/shell.h"
#include "permon/private/petsc/3.7/mat/transm.h"
#include "permon/private/petsc/3.7/pc/redundant.h"
#else
#error "unsupported PETSc version"
#endif

#endif
