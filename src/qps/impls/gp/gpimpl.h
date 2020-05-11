#if !defined(__GPIMPL_H)
#define __GPIMPL_H
#include <permon/private/qpsimpl.h>

typedef struct {
  PetscReal alpha;
  PetscReal alpha_user;
  QPSScalarArgType alpha_type;
  PetscReal gamma;
  PetscReal maxeig;
  PetscReal maxeig_tol;
  PetscInt  maxeig_iter;
  PetscReal btol;
  PetscReal bchop_tol;

  PetscReal gfnorm;
  PetscReal gcnorm;

  PetscInt  nmv;              /* ... matrix-vector mult. counter      */
  PetscInt  ncg;              /* ... cg step counter                  */
  PetscInt  nprop;            /* ... proportional step counter        */
  PetscInt  nexp;             /* ... expansion step counter           */
  PetscInt  nfinc;            /* ... functional increase counter      */
  PetscInt  nfall;            /* ... fallback step counter            */
  char      currentStepType;

  QPSMPGPExpansionType       exptype;
  QPSMPGPExpansionLengthType explengthtype;
  PetscErrorCode             (*expansion)(QPS,PetscReal,PetscReal);
  Vec                        expdirection;
  Vec                        explengthvec;
  Vec                        explengthvecold;
  Vec                        xold;
  PetscBool                  expproject;
  PetscBool                  resetalpha;
  PetscBool                  fallback;
  PetscBool                  fallback2;
  
  /* line search */
  PetscErrorCode (*gplinesearch)(QPS); /*                                */
  PetscReal      *ls_f;                /* cost function value array      */
  PetscReal      ls_alpha;             /* step length                    */
  PetscReal      ls_beta;              /* multiplier for alpha increment */
  PetscReal      ls_gamma;             /* Armijo rule parametr           */
  PetscInt       ls_M;                 /* size of ls_f                   */

} QPS_GP;

#endif
