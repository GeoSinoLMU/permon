
#include <private/qpimpl.h>

PetscClassId  QP_CLASSID;

const char *QPScaleTypes[]={"none","norm2","multiplicity","QPScalType","QPPF_",0};

static PetscErrorCode QPSetFromOptions_Private(QP qp);

#define QPView_PrintObjectLoaded(v,obj,description) PetscViewerASCIIPrintf(v,"    %-32s %-16s %c\n", description, obj?((PetscObject)obj)->name:"", obj?'Y':'N')
#define QPView_Vec(v,x,iname) \
{\
  PetscReal max,min,norm;\
  PetscInt  imax,imin;\
  const char *name = (iname);\
  TRY( VecNorm(x,NORM_2,&norm) );\
  TRY( VecMax(x,&imax,&max) );\
  TRY( VecMin(x,&imin,&min) );\
  TRY( PetscViewerASCIIPrintf(v, "||%2s|| = %.8e    max(%2s) = %.2e = %2s(%d)    min(%2s) = %.2e = %2s(%d)    %x\n",name,norm,name,max,name,imax,name,min,name,imin,x) );\
}

#undef __FUNCT__
#define __FUNCT__ "QPInitializeInitialVector_Private"
static PetscErrorCode QPInitializeInitialVector_Private(QP qp)
{
  Vec xp, xc;

  PetscFunctionBegin;
  if (qp->x) PetscFunctionReturn(0);
  if (!qp->parent) {
    /* if no initial guess exists, just set it to a zero vector */
    TRY( MatCreateVecs(qp->A,&qp->x,NULL) );
    TRY( VecZeroEntries(qp->x) ); // TODO: is it in the feasible set?
    PetscFunctionReturn(0);
  }
  TRY( QPGetSolutionVector(qp->parent, &xp) );
  if (xp) {
    TRY( VecDuplicate(xp, &xc) );
    TRY( VecCopy(xp, xc) );
    TRY( QPSetInitialVector(qp, xc) );
    TRY( VecDestroy(&xc) );
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPAddChild"
PetscErrorCode QPAddChild(QP qp, QPDuplicateOption opt, QP *newchild)
{
  QP child;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp,QP_CLASSID,1);
  PetscValidPointer(newchild,2);
  TRY( QPDuplicate(qp,opt,&child) );
  qp->child = child;
  child->parent = qp;
  child->id = qp->id+1;
  if (qp->changeListener) TRY( (*qp->changeListener)(qp) );
  if (newchild) *newchild = child;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPRemoveChild"
PetscErrorCode QPRemoveChild(QP qp)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp,QP_CLASSID,1);
  if (!qp->child) PetscFunctionReturn(0);
  qp->child->parent = NULL;
  qp->child->postSolve = NULL;
  TRY( QPDestroy(&qp->child) );
  if (qp->changeListener) TRY( (*qp->changeListener)(qp) );
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "QPCreate"
/*@
QPCreate - create instance of quadratic programming problem

Parameters:
+ comm - MPI comm 
- qp_out - pointer to created QP 
@*/
PetscErrorCode QPCreate(MPI_Comm comm, QP *qp_out)
{
  QP               qp;

  PetscFunctionBegin;
  PetscValidPointer(qp_out,2);

#if !defined(PETSC_USE_DYNAMIC_LIBRARIES)
  TRY( QPInitializePackage() );
#endif

#if PETSC_VERSION_MINOR < 6
  TRY( PetscHeaderCreate(qp,_p_QP,0,QP_CLASSID,"QP","Quadratic Programming Problem","QP",comm,QPDestroy,QPView) );
#else
  TRY( PetscHeaderCreate(qp,QP_CLASSID,"QP","Quadratic Programming Problem","QP",comm,QPDestroy,QPView) );
#endif
  TRY( PetscObjectChangeTypeName((PetscObject)qp,"QP") );
  qp->A            = NULL;
  qp->R            = NULL;
  qp->b            = NULL;
  qp->BE           = NULL;
  qp->BE_nest_count= 0;
  qp->cE           = NULL;
  qp->lambda_E     = NULL;
  qp->Bt_lambda    = NULL;
  qp->BI           = NULL;
  qp->cI           = NULL;
  qp->lambda_I     = NULL;
  qp->B            = NULL;
  qp->c            = NULL;
  qp->lambda       = NULL;
  qp->lb           = NULL;
  qp->lambda_lb    = NULL;
  qp->ub           = NULL;
  qp->lambda_ub    = NULL;
  qp->x            = NULL;
  qp->xwork        = NULL;
  qp->pc           = NULL;
  qp->pf           = NULL;
  qp->child        = NULL;
  qp->parent       = NULL;
  qp->solved       = PETSC_FALSE;
  qp->setupcalled  = PETSC_FALSE;
  qp->setfromoptionscalled = 0;

  qp->changeListener         = NULL;
  qp->changeListenerCtx      = NULL;
  qp->postSolve              = NULL;
  qp->postSolveCtx           = NULL;
  qp->postSolveCtxDestroy    = NULL;

  qp->id = 0;
  qp->transform = NULL;
  qp->transform_name[0] = 0;

  /* initialize preconditioner */
  TRY( QPGetPC(qp,&qp->pc) );

  *qp_out = qp;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPDuplicate"
/*@
   QPDuplicate - Duplicate QP.

   Input Parameters:
+  qp1 - QP to duplicate
-  opt - either QP_DUPLICATE_DO_NOT_COPY or QP_DUPLICATE_COPY_POINTERS

   Output Parameter:
.  qp2 - duplicated QP
@*/
PetscErrorCode QPDuplicate(QP qp1,QPDuplicateOption opt,QP *qp2)
{
  QP qp2_;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp1,QP_CLASSID,1);
  PetscValidPointer(qp2,2);
  TRY( QPCreate(PetscObjectComm((PetscObject)qp1),&qp2_) );

  if (opt==QP_DUPLICATE_DO_NOT_COPY) {
    *qp2 = qp2_;
    PetscFunctionReturn(0);
  }

  TRY( QPSetBox(qp2_,qp1->lb,qp1->ub) );
  TRY( QPSetChangeListener(qp2_,qp1->changeListener) );
  TRY( QPSetChangeListenerContext(qp2_,qp1->changeListenerCtx) );
  TRY( QPSetEq(qp2_,qp1->BE,qp1->cE) );
  TRY( QPSetEqMultiplier(qp2_,qp1->lambda_E) );
  qp2_->BE_nest_count = qp1->BE_nest_count;
  TRY( QPSetIneq(qp2_,qp1->BI,qp1->cI) );
  TRY( QPSetIneqMultiplier(qp2_,qp1->lambda_I) );
  TRY( QPSetInitialVector(qp2_,qp1->x) );
  TRY( QPSetLowerBoundMultiplier(qp2_,qp1->lambda_lb) );
  TRY( QPSetOperator(qp2_,qp1->A) );
  TRY( QPSetOperatorNullSpace(qp2_,qp1->R) );
  TRY( QPSetPC(qp2_,qp1->pc) );
  TRY( QPSetQPPF(qp2_,qp1->pf) );
  TRY( QPSetRhs(qp2_,qp1->b) );
  TRY( QPSetUpperBoundMultiplier(qp2_,qp1->lambda_ub) );
  TRY( QPSetWorkVector(qp2_,qp1->xwork) );

  if (qp1->lambda)    TRY( PetscObjectReference((PetscObject)(qp2_->lambda = qp1->lambda)) );
  if (qp1->Bt_lambda) TRY( PetscObjectReference((PetscObject)(qp2_->Bt_lambda = qp1->Bt_lambda)) );
  *qp2 = qp2_;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPCompareEqMultiplierWithLeastSquare"
PetscErrorCode QPCompareEqMultiplierWithLeastSquare(QP qp,PetscReal *norm)
{
  Vec BEt_lambda = NULL;
  Vec BEt_lambda_LS;
  QP qp2;

  PetscFunctionBegin;
  if (!qp->BE) PetscFunctionReturn(0);

  TRY( QPCompute_BEt_lambda(qp,&BEt_lambda) );

  TRY( QPDuplicate(qp,QP_DUPLICATE_COPY_POINTERS,&qp2) );
  TRY( QPSetEqMultiplier(qp2,NULL) );
  TRY( QPComputeMissingEqMultiplier(qp2) );
  TRY( QPCompute_BEt_lambda(qp2,&BEt_lambda_LS) );

  /* compare lambda_E with least-square lambda_E */
  TRY( VecAXPY(BEt_lambda_LS,-1.0,BEt_lambda) );
  TRY( VecNorm(BEt_lambda_LS,NORM_2,norm) );

  TRY( VecDestroy(&BEt_lambda) );
  TRY( VecDestroy(&BEt_lambda_LS) );
  TRY( QPDestroy(&qp2) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPViewKKT"
/*@
   QPViewKKT - View how well are satisfied KKT conditions.

   Input Parameters:
+  qp - the QP
-  v - visualization context

.seealso: QPChainViewKKT()
@*/
PetscErrorCode QPViewKKT(QP qp,PetscViewer v)
{
  PetscReal   normb=0.0,norm=0.0,dot=0.0;
  Vec         x,b,cE,cI,lb,ub,r,o,t;
  Mat         A,BE,BI;
  PetscBool   flg=PETSC_FALSE,compare_lambda_E=PETSC_FALSE,avail=PETSC_TRUE;
  MPI_Comm    comm;
  char        *kkt_name;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp,QP_CLASSID,1);
  TRY( PetscObjectGetComm((PetscObject)qp,&comm) );
  if (!v) v = PETSC_VIEWER_STDOUT_(comm);
  PetscValidHeaderSpecific(v,PETSC_VIEWER_CLASSID,2);
  PetscCheckSameComm(qp,1,v,2);

  TRY( PetscObjectTypeCompare((PetscObject)v,PETSCVIEWERASCII,&flg) );
  if (!flg) FLLOP_SETERRQ1(comm,PETSC_ERR_SUP,"Viewer type %s not supported for QP",((PetscObject)v)->type_name);

  TRY( PetscOptionsGetBool(((PetscObject)qp)->options,NULL,"-qp_view_kkt_compare_lambda_E",&compare_lambda_E,NULL) );

  TRY( PetscObjectName((PetscObject)qp) );
  TRY( PetscObjectPrintClassNamePrefixType((PetscObject)qp,v) );
  TRY( PetscViewerASCIIPrintf(v, "  #%d in chain, derived by %s\n",qp->id,qp->transform_name) );
  if (!qp->solved) {
    TRY( PetscViewerASCIIPrintf(v, "*** WARNING: QP is not solved. ***\n") );
  }

  TRY( QPGetOperator(qp, &A) );
  TRY( QPGetRhs(qp, &b) );
  TRY( QPGetEq(qp, &BE, &cE) );
  TRY( QPGetIneq(qp, &BI, &cI) );
  TRY( QPGetBox(qp, &lb, &ub) );
  TRY( QPGetSolutionVector(qp, &x) );
  TRY( VecNorm(b,NORM_2,&normb) );

  QPView_Vec(v,x,"x");
  QPView_Vec(v,b,"b");
  if (cE) QPView_Vec(v,cE,"cE");
  if (BE && !cE) TRY( PetscViewerASCIIPrintf(v, "||cE|| = 0.00e-00    max(cE) = 0.00e-00 = cE(0)    min(cE) = 0.00e-00 = cE(0)\n") );
  if (cI) QPView_Vec(v,cI,"cI");
  if (BI && !cI) TRY( PetscViewerASCIIPrintf(v, "||cI|| = 0.00e-00    max(cI) = 0.00e-00 = cI(0)    min(cI) = 0.00e-00 = cI(0)\n") );
  if (lb) {
    IS isf;
    Vec lbf;
    QPView_Vec(v,lb,"lb");
    TRY( PetscObjectQuery((PetscObject)lb,"is_finite",(PetscObject*)&isf) );
    if (isf) {
      TRY( PetscInfo(qp,"is_finite composed to lb found\n") );
      TRY( VecGetSubVector(lb,isf,&lbf) );
      QPView_Vec(v,lbf,"lf");
      TRY( VecRestoreSubVector(lb,isf,&lbf) );
    } else {
      PetscInt i,m;
      PetscScalar *a;
      TRY( VecGetLocalSize(lb,&m) );
      TRY( VecDuplicate(lb,&lbf) );
      TRY( VecCopy(lb,lbf) );
      TRY( VecGetArray(lbf,&a) );
      for (i=0; i<m; i++) {
        if (a[i] <= PETSC_NINFINITY) a[i]=0.0;
      }
      TRY( VecRestoreArray(lbf,&a) );
      QPView_Vec(v,lbf,"lf");
      TRY( VecDestroy(&lbf) );
    }
  }
  if (ub) {
    Vec ubf;
    PetscInt i,m;
    PetscScalar *a;
    TRY( VecGetLocalSize(ub,&m) );
    TRY( VecDuplicate(ub,&ubf) );
    TRY( VecCopy(ub,ubf) );
    TRY( VecGetArray(ubf,&a) );
    for (i=0; i<m; i++) {
      if (a[i] >= PETSC_INFINITY) a[i]=0.0;
    }
    TRY( VecRestoreArray(ubf,&a) );
    QPView_Vec(v,ub,"ub");
    QPView_Vec(v,ubf,"uf");
    TRY( VecDestroy(&ubf) );
  }
  
  TRY( VecDuplicate(b, &r) );
  TRY( QPComputeLagrangianGradient(qp,x,r,&kkt_name) );
  TRY( VecIsValid(r,&avail) );

  if (avail) {
    if (compare_lambda_E) {
      TRY( QPCompareEqMultiplierWithLeastSquare(qp,&norm) );
      TRY( PetscViewerASCIIPrintf(v,"||BE'*lambda_E - BE'*lambda_E_LS|| = %.4e\n",norm) );
    }
    TRY( VecNorm(r, NORM_2, &norm) );
    TRY( PetscViewerASCIIPrintf(v,"r = ||%s|| = %.2e    rO/||b|| = %.2e\n",kkt_name,norm,norm/normb) );
  } else {
    TRY( PetscViewerASCIIPrintf(v,"r = ||%s|| not available\n",kkt_name) );
  }
  TRY( VecDestroy(&r) );

  if (BE) {
    if (BE->ops->mult) {
      TRY( MatCreateVecs(BE, NULL, &r) );
      TRY( MatMult(BE, x, r) );
      if (cE) TRY( VecAXPY(r, -1.0, cE) );
      TRY( VecNorm(r, NORM_2, &norm) );
      if (cE) {
        TRY( PetscViewerASCIIPrintf(v,"r = ||BE*x-cE||          = %.2e    r/||b|| = %.2e\n",norm,norm/normb) );
      } else {
        TRY( PetscViewerASCIIPrintf(v,"r = ||BE*x||             = %.2e    r/||b|| = %.2e\n",norm,norm/normb) );
      }
      TRY( VecDestroy(&r) );
    } else {
      if (cE) {
        TRY( PetscViewerASCIIPrintf(v,"r = ||BE*x-cE||         not available\n") );
      } else {
        Vec t = qp->xwork;
        TRY( QPPFApplyGtG(qp->pf,x,t) );                    /* BEtBEx = BE'*BE*x */
        TRY( VecDot(x,t,&norm) );                           /* norm = x'*BE'*BE*x */
        norm = PetscSqrtReal(norm);
        TRY( PetscViewerASCIIPrintf(v,"r = ||BE*x||             = %.2e    r/||b|| = %.2e\n",norm,norm/normb) );
      }
    }
  }

  if (BI) {
    TRY( VecDuplicate(qp->lambda_I,&r) );
    TRY( VecDuplicate(r,&o) );
    TRY( VecDuplicate(r,&t) );

    TRY( VecSet(o,0.0) );                                   /* o = zeros(size(r)) */

    /* r = BI*x - cI */
    TRY( MatMult(BI, x, r) );                               /* r = BI*x         */
    if (cI) TRY( VecAXPY(r, -1.0, cI) );                    /* r = r - cI       */

    /* rI = norm(max(BI*x-cI,0)) */
    TRY( VecPointwiseMax(t,r,o) );                          /* t = max(r,o)     */
    TRY( VecNorm(t,NORM_2,&norm) );                         /* norm = norm(t)     */
    if (cI) {
      TRY( PetscViewerASCIIPrintf(v,"r = ||max(BI*x-cI,0)||   = %.2e    r/||b|| = %.2e\n",norm,norm/normb) );
    } else {
      TRY( PetscViewerASCIIPrintf(v,"r = ||max(BI*x,0)||      = %.2e    r/||b|| = %.2e\n",norm,norm/normb) );
    }

    /* lambda >= o  =>  examine min(lambda,o) */
    TRY( VecSet(o,0.0) );                                   /* o = zeros(size(r)) */
    TRY( VecPointwiseMin(t,qp->lambda_I,o) );
    TRY( VecNorm(t,NORM_2,&norm) );                         /* norm = ||min(lambda,o)|| */
    TRY( PetscViewerASCIIPrintf(v,"r = ||min(lambda_I,0)||  = %.2e    r/||b|| = %.2e\n",norm,norm/normb) );

    /* lambda'*(BI*x-cI) = 0 */
    TRY( VecDot(qp->lambda_I,r,&dot) );
    dot = PetscAbs(dot);
    if (cI) {
      TRY( PetscViewerASCIIPrintf(v,"r = |lambda_I'*(BI*x-cI)|= %.2e    r/||b|| = %.2e\n",dot,dot/normb) );
    } else {
      TRY( PetscViewerASCIIPrintf(v,"r = |lambda_I'*(BI*x)|= %.2e       r/||b|| = %.2e\n",dot,dot/normb) );
    }

    TRY( VecDestroy(&o) );
    TRY( VecDestroy(&r) );
    TRY( VecDestroy(&t) );
  }

  if (lb) {
    TRY( VecDuplicate(x,&o) );
    TRY( VecDuplicate(x,&r) );

    /* rI = norm(min(x-lb,0)) */
    TRY( VecSet(o,0.0) );                                   /* o = zeros(size(r)) */
    TRY( VecWAXPY(r, -1.0, lb, x) );                        /* r = x - lb       */
    TRY( VecPointwiseMin(r,r,o) );                          /* r = min(r,o)     */
    TRY( VecNorm(r,NORM_2,&norm) );                         /* norm = norm(r)     */
    TRY( PetscViewerASCIIPrintf(v,"r = ||min(x-lb,0)||      = %.2e    r/||b|| = %.2e\n",norm,norm/normb) );

    /* lambda >= o  =>  examine min(lambda,o) */
    TRY( VecSet(o,0.0) );                                   /* o = zeros(size(r)) */
    TRY( VecPointwiseMin(r,qp->lambda_lb,o) );
    TRY( VecNorm(r,NORM_2,&norm) );                         /* norm = ||min(lambda,o)|| */
    TRY( PetscViewerASCIIPrintf(v,"r = ||min(lambda_lb,0)|| = %.2e    r/||b|| = %.2e\n",norm,norm/normb) );

    /* lambda'*(lb-x) = 0 */
    TRY( VecCopy(lb,r) );
    TRY( VecAXPY(r,-1.0,x) );
    {
      PetscInt i,n;
      PetscScalar *rarr;
      const PetscScalar *larr;
      TRY( VecGetLocalSize(r,&n) );
      TRY( VecGetArray(r,&rarr) );
      TRY( VecGetArrayRead(lb,&larr) );
      for (i=0; i<n; i++) if (larr[i]<=PETSC_NINFINITY) rarr[i]=-1.0;
      TRY( VecRestoreArray(r,&rarr) );
      TRY( VecRestoreArrayRead(lb,&larr) );
    }
    TRY( VecDot(qp->lambda_lb,r,&dot) );
    dot = PetscAbs(dot);
    TRY( PetscViewerASCIIPrintf(v,"r = |lambda_lb'*(lb-x)|  = %.2e    r/||b|| = %.2e\n",dot,dot/normb) );

    TRY( VecDestroy(&o) );
    TRY( VecDestroy(&r) );
  }

  if (ub) {
    TRY( VecDuplicate(x,&o) );
    TRY( VecDuplicate(x,&r) );

    /* rI = norm(max(x-ub,0)) */
    TRY( VecDuplicate(x,&r) );
    TRY( VecDuplicate(x,&o) );
    TRY( VecSet(o,0.0) );                                   /* o = zeros(size(r)) */
    TRY( VecWAXPY(r, -1.0, ub, x) );                        /* r = x - ub       */
    TRY( VecPointwiseMax(r,r,o) );                          /* r = max(r,o)     */
    TRY( VecNorm(r,NORM_2,&norm) );                         /* norm = norm(r)     */
    TRY( PetscViewerASCIIPrintf(v,"r = ||max(x-ub,0)||      = %.2e    r/||b|| = %.2e\n",norm,norm/normb) );

    /* lambda >= o  =>  examine min(lambda,o) */
    TRY( VecSet(o,0.0) );                                   /* o = zeros(size(r)) */
    TRY( VecPointwiseMin(r,qp->lambda_ub,o) );
    TRY( VecNorm(r,NORM_2,&norm) );                         /* norm = ||min(lambda,o)|| */
    TRY( PetscViewerASCIIPrintf(v,"r = ||min(lambda_ub,0)|| = %.2e    r/||b|| = %.2e\n",norm,norm/normb) );

    /* lambda'*(x-ub) = 0 */
    TRY( VecCopy(ub,r) );
    TRY( VecAYPX(r,-1.0,x) );
    {
      PetscInt i,n;
      PetscScalar *rarr;
      const PetscScalar *uarr;
      TRY( VecGetLocalSize(r,&n) );
      TRY( VecGetArray(r,&rarr) );
      TRY( VecGetArrayRead(ub,&uarr) );
      for (i=0; i<n; i++) if (uarr[i]>=PETSC_INFINITY) rarr[i]=1.0;
      TRY( VecRestoreArray(r,&rarr) );
      TRY( VecRestoreArrayRead(ub,&uarr) );
    }
    TRY( VecDot(qp->lambda_ub,r,&dot) );
    dot = PetscAbs(dot);
    TRY( PetscViewerASCIIPrintf(v,"r = |lambda_ub'*(x-ub)|  = %.2e    r/||b|| = %.2e\n",dot,dot/normb) );

    TRY( VecDestroy(&o) );
    TRY( VecDestroy(&r) );
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPView"
/*@
   QPView - View information about QP.

   Input Parameters:
+  qp - the QP
-  v - visualization context
@*/
PetscErrorCode QPView(QP qp,PetscViewer v)
{
  Vec         b,cE,cI,lb,ub;
  Mat         A,R,BE,BI;
  PetscBool   iascii;
  MPI_Comm    comm;
  QP          childDual;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp,QP_CLASSID,1);
  TRY( PetscObjectGetComm((PetscObject)qp,&comm) );
  if (!v) v = PETSC_VIEWER_STDOUT_(comm);
  PetscValidHeaderSpecific(v,PETSC_VIEWER_CLASSID,2);
  PetscCheckSameComm(qp,1,v,2);

  TRY( PetscObjectTypeCompare((PetscObject)v,PETSCVIEWERASCII,&iascii) );
  if (!iascii) FLLOP_SETERRQ1(comm,PETSC_ERR_SUP,"Viewer type %s not supported for QP",((PetscObject)v)->type_name);
  TRY( PetscObjectName((PetscObject)qp) );
  TRY( PetscObjectPrintClassNamePrefixType((PetscObject)qp,v) );
  TRY( PetscViewerASCIIPrintf(v, "#%d in chain, derived by %s\n",qp->id,qp->transform_name) );

  TRY( QPGetOperator(qp, &A) );
  TRY( QPGetOperatorNullSpace(qp, &R) );
  TRY( QPGetRhs(qp, &b) );
  TRY( QPGetBox(qp, &lb, &ub) );
  TRY( QPGetEq(qp, &BE, &cE) );
  TRY( QPGetIneq(qp, &BI, &cI) );
  TRY( QPChainFind(qp, (PetscErrorCode(*)(QP))QPTDualize, &childDual) );

  TRY( PetscViewerASCIIPrintf(v,"  LOADED OBJECTS:\n") );
  TRY( PetscViewerASCIIPrintf(v,"    %-32s %-16s %s\n", "what", "name", "present") );
  TRY( QPView_PrintObjectLoaded(v, A,   "Hessian") );
  TRY( QPView_PrintObjectLoaded(v, b,   "linear term (right-hand-side)") );
  TRY( QPView_PrintObjectLoaded(v, R,   "R (kernel of K)") );
  TRY( QPView_PrintObjectLoaded(v, lb,  "lower bounds") );
  TRY( QPView_PrintObjectLoaded(v, ub,  "upper bounds") );
  TRY( QPView_PrintObjectLoaded(v, BE,  "linear eq. constraint matrix") );
  TRY( QPView_PrintObjectLoaded(v, cE,  "linear eq. constraint RHS") );
  TRY( QPView_PrintObjectLoaded(v, BI,  "linear ineq. constraint") );
  TRY( QPView_PrintObjectLoaded(v, cI,  "linear ineq. constraint RHS") );

  //TODO print FETI flag
  if (A)   TRY( MatPrintInfo(A) );
  if (b)   TRY( VecPrintInfo(b) );
  if (R)   TRY( MatPrintInfo(R) );
  if (lb)  TRY( VecPrintInfo(lb) );
  if (ub)  TRY( VecPrintInfo(ub) );
  if (BE)  TRY( MatPrintInfo(BE) );
  if (cE)  TRY( VecPrintInfo(cE) );
  if (BI)  TRY( MatPrintInfo(BI) );
  if (cI)  TRY( VecPrintInfo(cI) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPReset"
/*@
   QPReset - Resets a QP context to the QPsetupcalled = 0 state, destroys child, PC, Vecs,  Mats, etc.

   Collective on QP

   Input Parameter:
.  qp - the QP
@*/
PetscErrorCode QPReset(QP qp)
{
  PetscFunctionBegin;
  TRY( QPDestroy( &qp->child) );

  TRY( MatDestroy(&qp->A) );
  TRY( MatDestroy(&qp->R) );
  TRY( MatDestroy(&qp->BE) );
  TRY( MatDestroy(&qp->BI) );
  TRY( MatDestroy(&qp->B) );

  TRY( VecDestroy(&qp->b) );
  TRY( VecDestroy(&qp->x) );
  TRY( VecDestroy(&qp->xwork) );
  TRY( VecDestroy(&qp->cE) );
  TRY( VecDestroy(&qp->lambda_E) );
  TRY( VecDestroy(&qp->Bt_lambda) );
  TRY( VecDestroy(&qp->cI) );
  TRY( VecDestroy(&qp->lambda_I) );
  TRY( VecDestroy(&qp->c) );
  TRY( VecDestroy(&qp->lambda) );
  TRY( VecDestroy(&qp->lb) );
  TRY( VecDestroy(&qp->lambda_lb) );
  TRY( VecDestroy(&qp->ub) );
  TRY( VecDestroy(&qp->lambda_ub) );
  
  TRY( PCDestroy( &qp->pc) );

  TRY( QPPFDestroy(&qp->pf) );
  qp->setupcalled = PETSC_FALSE;
  qp->solved = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPSetUpInnerObjects"
PetscErrorCode QPSetUpInnerObjects(QP qp)
{
  MPI_Comm comm;
  PetscInt i;
  Mat Bs[2];
  IS rows[2];
  Vec cs[2],c[2];
  Vec lambdas[2];

  FllopTracedFunctionBegin;
  PetscValidHeaderSpecific(qp,QP_CLASSID,1);

  TRY( PetscObjectGetComm((PetscObject)qp,&comm) );
  if (!qp->A) FLLOP_SETERRQ(comm,PETSC_ERR_ORDER,"Hessian must be set before " __FUNCT__);
  if (!qp->b) FLLOP_SETERRQ(comm,PETSC_ERR_ORDER,"linear term must be set before " __FUNCT__);

  FllopTraceBegin;
  TRY( PetscInfo1(qp,"setup inner objects for QP #%d\n",qp->id) );

  TRY( QPInitializeInitialVector_Private(qp) );

  if (!qp->xwork) TRY( VecDuplicate(qp->x,&qp->xwork) );

  if (qp->lb && !qp->lambda_lb) {
    TRY( VecDuplicate(qp->b,&qp->lambda_lb) );
    TRY( VecInvalidate(qp->lambda_lb) );
  }
  if (!qp->lb) {
    TRY( VecDestroy(&qp->lambda_lb) );
  }

  if (qp->ub && !qp->lambda_ub) {
    TRY( VecDuplicate(qp->b,&qp->lambda_ub) );
    TRY( VecInvalidate(qp->lambda_ub) );
  }
  if (!qp->ub) {
    TRY( VecDestroy(&qp->lambda_ub) );
  }

  if (qp->BE && !qp->lambda_E) {
    TRY( MatCreateVecs(qp->BE,NULL,&qp->lambda_E) );
    TRY( VecInvalidate(qp->lambda_E) );
  }
  if (!qp->BE) {
    TRY( VecDestroy(&qp->lambda_E) );
  }

  if (qp->BI && !qp->lambda_I) {
    TRY( MatCreateVecs(qp->BI,NULL,&qp->lambda_I) );
    TRY( VecInvalidate(qp->lambda_I) );
  }
  if (!qp->BI) {
    TRY( VecDestroy(&qp->lambda_I) );
  }

  if ((qp->BE || qp->BI) && !qp->B)
  {
  TRY( VecDestroy(&qp->c) );

  if (qp->BE && !qp->BI) {
    TRY( PetscObjectReference((PetscObject)(qp->B       = qp->BE)) );
    TRY( PetscObjectReference((PetscObject)(qp->lambda  = qp->lambda_E)) );
    TRY( PetscObjectReference((PetscObject)(qp->c       = qp->cE)) );
  } else if (!qp->BE && qp->BI) {
    TRY( PetscObjectReference((PetscObject)(qp->B       = qp->BI)) );
    TRY( PetscObjectReference((PetscObject)(qp->lambda  = qp->lambda_I)) );
    TRY( PetscObjectReference((PetscObject)(qp->c       = qp->cI)) );
  } else {
    TRY( PetscObjectReference((PetscObject)(Bs[0]       = qp->BE)) );
    TRY( PetscObjectReference((PetscObject)(lambdas[0]  = qp->lambda_E)) );
    if (qp->cE) {
      TRY( PetscObjectReference((PetscObject)(cs[0]     = qp->cE)) );
    } else {
      TRY( MatCreateVecs(Bs[0],NULL,&cs[0]) );
      TRY( VecSet(cs[0],0.0) );
    }
    
    TRY( PetscObjectReference((PetscObject)(Bs[1]       = qp->BI)) );
    TRY( PetscObjectReference((PetscObject)(lambdas[1]  = qp->lambda_I)) );
    if (qp->cI) {
      TRY( PetscObjectReference((PetscObject)(cs[1]     = qp->cI)) );
    } else {
      TRY( MatCreateVecs(Bs[1],NULL,&cs[1]) );
      TRY( VecSet(cs[1],0.0) );
    }
    
    TRY( MatCreateNestPermon(comm,2,NULL,1,NULL,Bs,&qp->B) );
    TRY( MatCreateVecs(qp->B,NULL,&qp->c) );
    TRY( PetscObjectSetName((PetscObject)qp->B,"B") );
    TRY( PetscObjectSetName((PetscObject)qp->c,"c") );
    
    /* copy cE,cI to c */
    TRY( MatNestGetISs(qp->B,rows,NULL) );
    for (i=0; i<2; i++) {
      TRY( VecGetSubVector(qp->c,rows[i],&c[i]) );
      TRY( VecCopy(cs[i],c[i]) );
      TRY( VecRestoreSubVector(qp->c,rows[i],&c[i]) );
    }
    
    for (i=0; i<2; i++) {
      TRY( MatDestroy(&Bs[i]) );
      TRY( VecDestroy(&cs[i]) );
      TRY( VecDestroy(&lambdas[i]) );
    }
  }
  }

  if (qp->B && !qp->lambda) {
    TRY( MatCreateVecs(qp->B,NULL,&qp->lambda) );
    TRY( PetscObjectSetName((PetscObject)qp->lambda,"lambda") );
    TRY( VecInvalidate(qp->lambda) );
  }

  if (qp->B && !qp->Bt_lambda) {
    TRY( MatCreateVecs(qp->B,&qp->Bt_lambda,NULL) );
    TRY( PetscObjectSetName((PetscObject)qp->lambda,"Bt_lambda") );
    TRY( VecInvalidate(qp->Bt_lambda) );
  }

  if (!qp->B) {
    TRY( VecDestroy(&qp->lambda) );
    TRY( VecDestroy(&qp->Bt_lambda) );
  }
  PetscFunctionReturnI(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPSetUp"
/*@
   QPSetUp - Sets up the internal data structures for the QP.

   Collective on QP

   Input Parameter:
.  qp   - the QP
@*/
PetscErrorCode QPSetUp(QP qp)
{
  MPI_Comm comm;

  FllopTracedFunctionBegin;
  PetscValidHeaderSpecific(qp,QP_CLASSID,1);
  if (qp->setupcalled) PetscFunctionReturn(0);

  TRY( PetscObjectGetComm((PetscObject)qp,&comm) );
  if (!qp->A) FLLOP_SETERRQ(comm,PETSC_ERR_ORDER,"Hessian must be set before " __FUNCT__);
  if (!qp->b) FLLOP_SETERRQ(comm,PETSC_ERR_ORDER,"linear term must be set before " __FUNCT__);

  FllopTraceBegin;
  TRY( PetscInfo1(qp,"setup QP #%d\n",qp->id) );
  TRY( QPSetUpInnerObjects(qp) );
  if (!qp->pc) TRY( QPGetPC(qp,&qp->pc) );
  TRY( PCSetOperators(qp->pc,qp->A,qp->A) );
  TRY( QPSetFromOptions_Private(qp) );
  TRY( PCSetUp(qp->pc) );
  qp->setupcalled = PETSC_TRUE;
  PetscFunctionReturnI(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPCheckNullSpace"
PetscErrorCode QPCheckNullSpace(QP qp,PetscReal tol)
{
  Mat K,R;
  Vec d,x,y;
  PetscReal normd,normy;

  PetscFunctionBeginI;
  TRY( QPGetOperator(qp,&K) );
  TRY( QPGetOperatorNullSpace(qp,&R) );
  TRY( MatCreateVecs(K,&d,&y) );
  TRY( MatCreateVecs(R,&x,NULL) );
  TRY( MatGetDiagonal(K,d) );
  TRY( VecNorm(d,NORM_2,&normd) );
  TRY( VecSetRandom(x,NULL) );
  TRY( MatMult(R,x,d) );
  TRY( MatMult(K,d,y) );
  TRY( VecNorm(y,NORM_2,&normy) );
  TRY( PetscInfo3(fllop,"||K*R*x|| = %.3e   ||diag(K)|| = %.3e    ||K*R*x|| / ||diag(K)|| = %.3e\n",normy,normd,normy/normd) );
  FLLOP_ASSERT1(normy / normd < tol, "||K*R*x|| / ||diag(K)|| < %.1e", tol);
  TRY( VecDestroy(&d) );
  TRY( VecDestroy(&x) );
  TRY( VecDestroy(&y) );
  PetscFunctionReturnI(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPCompute_BEt_lambda"
PetscErrorCode QPCompute_BEt_lambda(QP qp,Vec *BEt_lambda)
{
  PetscBool   flg=PETSC_FALSE;

  PetscFunctionBegin;
  *BEt_lambda = NULL;
  if (!qp->BE) PetscFunctionReturn(0);

  if (!qp->BI) {
    TRY( VecIsInvalidated(qp->Bt_lambda, &flg) );
    if (!flg) {
      TRY( VecDuplicate(qp->Bt_lambda, BEt_lambda) );
      TRY( VecCopy(qp->Bt_lambda, *BEt_lambda) );                               /* BEt_lambda = Bt_lambda */
      PetscFunctionReturn(0);
    }

    TRY( VecIsInvalidated(qp->lambda,&flg) );
    if (!flg && qp->B->ops->multtranspose) {
      TRY( MatCreateVecs(qp->B, BEt_lambda, NULL) );
      TRY( MatMultTranspose(qp->B, qp->lambda, *BEt_lambda) );                  /* BEt_lambda = B'*lambda */
      PetscFunctionReturn(0);
    }
  }

  TRY( VecIsInvalidated(qp->lambda_E,&flg) );
  if (!flg || !qp->BE->ops->multtranspose) {
    TRY( MatCreateVecs(qp->BE, BEt_lambda, NULL) );
    TRY( MatMultTranspose(qp->BE, qp->lambda_E, *BEt_lambda) );                 /* Bt_lambda = BE'*lambda_E */
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPComputeLagrangianGradient"
PetscErrorCode QPComputeLagrangianGradient(QP qp, Vec x, Vec r, char *kkt_name_[])
{
  Vec         b,cE,cI,lb,ub,Bt_lambda=NULL,BEt_lambda=NULL;
  Mat         A,BE,BI;
  PetscBool   flg=PETSC_FALSE,avail=PETSC_TRUE;
  char        kkt_name[256]="A*x - b";

  PetscFunctionBegin;
  TRY( QPSetUp(qp) );
  TRY( QPGetOperator(qp, &A) );
  TRY( QPGetRhs(qp, &b) );
  TRY( QPGetEq(qp, &BE, &cE) );
  TRY( QPGetIneq(qp, &BI, &cI) );
  TRY( QPGetBox(qp, &lb, &ub) );

  TRY( MatMult(A, x, r) );
  TRY( VecAXPY(r, -1.0, b) );                                                   /* r = A*x - b */

  if (qp->lb) {
    TRY( VecAXPY(r,-1.0,qp->lambda_lb) );                                       /* r = r - lambda_lb */
  }
  if (qp->ub) {
    TRY( VecAXPY(r, 1.0,qp->lambda_ub) );                                       /* r = r + lambda_ub */
  }

  TRY( VecDuplicate(r,&Bt_lambda) );
  if (qp->B) {
    TRY( VecIsInvalidated(qp->Bt_lambda,&flg) );
    if (!flg) {
      TRY( VecCopy(qp->Bt_lambda,Bt_lambda) );                                  /* Bt_lambda = (B'*lambda) */
      if (kkt_name_) TRY( PetscStrcat(kkt_name," + (B'*lambda)") );
      if (!qp->BI) TRY( PetscObjectReference((PetscObject)(BEt_lambda = Bt_lambda)) );
      goto endif;
    }

    TRY( VecIsInvalidated(qp->lambda,&flg) );
    if (!flg && qp->B->ops->multtranspose) {
      TRY( MatMultTranspose(qp->B, qp->lambda, Bt_lambda) );                    /* Bt_lambda = B'*lambda */
      if (kkt_name_) TRY( PetscStrcat(kkt_name," + B'*lambda") );
      if (!qp->BI) TRY( PetscObjectReference((PetscObject)(BEt_lambda = Bt_lambda)) );
      goto endif;
    }

    if (qp->BE) {
      if (kkt_name_) TRY( PetscStrcat(kkt_name," + BE'*lambda_E") );
    }
    if (qp->BI) {
      if (kkt_name_) TRY( PetscStrcat(kkt_name," + BI'*lambda_I") );
    }

    if (qp->BE) {
      TRY( VecIsInvalidated(qp->lambda_E,&flg) );
      if (flg || !qp->BE->ops->multtranspose) {
        avail = PETSC_FALSE;
        goto endif;
      }
      TRY( MatMultTranspose(BE, qp->lambda_E, Bt_lambda) );                     /* Bt_lambda = BE'*lambda_E */
    }

    if (qp->BI) {
      TRY( VecIsInvalidated(qp->lambda_I,&flg) );
      if (flg || !qp->BI->ops->multtransposeadd) {
        avail = PETSC_FALSE;
        goto endif;
      }
      TRY( MatMultTransposeAdd(BI, qp->lambda_I, Bt_lambda, Bt_lambda) );       /* Bt_lambda = Bt_lambda + BI'*lambda_I */
    } else {
      TRY( PetscObjectReference((PetscObject)(BEt_lambda = Bt_lambda)) );
    }
  }
  endif:
  if (avail) {
    TRY( VecAXPY(r,1.0,Bt_lambda) );                                            /* r = r + Bt_lambda */
    if (qp->lb) {
      if (kkt_name_) TRY( PetscStrcat(kkt_name," - lambda_lb") );
    }
    if (qp->ub) {
      if (kkt_name_) TRY( PetscStrcat(kkt_name," + lambda_ub") );
    }
  } else {
    TRY( VecInvalidate(r) );
  }
  if (kkt_name_) TRY( PetscStrallocpy(kkt_name,kkt_name_) );
  TRY( VecDestroy(&Bt_lambda) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPComputeMissingEqMultiplier"
PetscErrorCode QPComputeMissingEqMultiplier(QP qp)
{
  Vec r = qp->xwork;
  PetscBool flg;
  const char *name;
  QP qp_I;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp,QP_CLASSID,1);
  TRY( QPSetUp(qp) );

  if (!qp->BE) PetscFunctionReturn(0);
  TRY( VecIsInvalidated(qp->lambda_E,&flg) );
  if (!flg) PetscFunctionReturn(0);
  if (qp->Bt_lambda) {
    TRY( VecIsInvalidated(qp->Bt_lambda,&flg) );
    if (!flg) PetscFunctionReturn(0);
  }

  TRY( QPDuplicate(qp,QP_DUPLICATE_COPY_POINTERS,&qp_I) );
  TRY( QPSetEq(qp_I,NULL,NULL) );
  TRY( QPComputeLagrangianGradient(qp_I,qp->x,r,NULL) );
  TRY( QPDestroy(&qp_I) );

  //TODO Should we add qp->Bt_lambda_E ?
  if (qp->BE == qp->B) {
    TRY( VecCopy(r,qp->Bt_lambda) );
    TRY( VecScale(qp->Bt_lambda,-1.0) );
  } else {
    TRY( QPPFApplyHalfQ(qp->pf,r,qp->lambda_E) );
    TRY( VecScale(qp->lambda_E,-1.0) );                                               /* lambda_E_LS = -(BE*BE')\\BE*r */
  }

  if (FllopDebugEnabled) {
    PetscReal norm;
    if (qp->BE->ops->multtranspose) {
      TRY( MatMultTransposeAdd(qp->BE,qp->lambda_E,r,r) );
    } else {
      TRY( VecAXPY(r,1.0,qp->Bt_lambda) );
    }
    TRY( VecNorm(r,NORM_2,&norm) );
    TRY( FllopDebug1("||r||=%.2e\n",norm) );
  }

  TRY( PetscObjectGetName((PetscObject)qp,&name) );
  TRY( PetscInfo3(qp,"missing eq. con. multiplier computed for QP Object %s (#%d in chain, derived by %s)\n",name,qp->id,qp->transform_name) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPComputeMissingBoxMultipliers"
PetscErrorCode QPComputeMissingBoxMultipliers(QP qp)
{
  Vec lb = qp->lb;
  Vec ub = qp->ub;
  Vec lambda_lb = qp->lambda_lb;
  Vec lambda_ub = qp->lambda_ub;
  Vec r = qp->xwork;
  PetscBool flg=PETSC_TRUE,flg2=PETSC_TRUE;
  QP qp_E;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp,QP_CLASSID,1);
  TRY( QPSetUp(qp) );

  if (lb) {
    TRY( VecIsValid(lambda_lb,&flg) );
  }
  if (ub) {
    TRY( VecIsValid(lambda_ub,&flg2) );
  }
  if (!lb && !ub) PetscFunctionReturn(0);
  if (flg && flg2) PetscFunctionReturn(0);

  /* currently cannot handle this situation, leave multipliers untouched */
  if (qp->BI) PetscFunctionReturn(0);

  TRY( QPDuplicate(qp,QP_DUPLICATE_COPY_POINTERS,&qp_E) );
  TRY( QPSetBox(qp_E,NULL,NULL) );
  TRY( QPComputeLagrangianGradient(qp_E,qp->x,r,NULL) );
  TRY( QPDestroy(&qp_E) );

  if (lb) {
    TRY( VecCopy(r,lambda_lb) );
  }
  if (ub) {
    TRY( VecCopy(r,lambda_ub) );
    TRY( VecScale(lambda_ub,-1.0) );
  }
  if (lb && ub) {
    TRY( VecZeroEntries(r) );
    TRY( VecPointwiseMax(lambda_lb,lambda_lb,r) );
    TRY( VecPointwiseMax(lambda_ub,lambda_ub,r) );
  }

  {
    const char *name_qp,*name_lambda;
    TRY( PetscObjectGetName((PetscObject)qp,&name_qp) );
    TRY( PetscObjectGetName((PetscObject)qp->lambda_lb,&name_lambda) );
    TRY( PetscInfo4(qp,"missing lower bound con. multiplier %s computed for QP Object %s (#%d in chain, derived by %s)\n",name_lambda,name_qp,qp->id,qp->transform_name) );
  }
  PetscFunctionReturn(0);
}

//TODO remove once MPGP fully supports ub=NULL
#undef __FUNCT__
#define __FUNCT__ "QPRemoveInactiveBounds"
PetscErrorCode QPRemoveInactiveBounds(QP qp)
{
  PetscReal extrem;

  PetscFunctionBegin;
  if (qp->ub) {
    TRY( VecMin(qp->ub,NULL,&extrem) );
    TRY( PetscInfo1(qp, "min(ub) = %0.2e\n", extrem) );
    if (extrem >= PETSC_INFINITY) {
      TRY( PetscInfo1(qp, "inactive upper bound constraint detected\n", extrem) );
      TRY( VecDestroy(&qp->ub) );
    }
  }
  if (qp->lb) {
    TRY( VecMax(qp->lb,NULL,&extrem) );
    TRY( PetscInfo1(qp, "max(lb) = %0.2e\n", extrem) );
    if (extrem <= PETSC_NINFINITY) {
      TRY( PetscInfo1(qp, "inactive lower bound constraint detected\n", extrem) );
      TRY( VecDestroy(&qp->lb) );
    }
  }
  if (!qp->ub && qp->lambda_ub) {
    TRY( VecDestroy(&qp->lambda_ub) );
  }
  if (!qp->lb && qp->lambda_lb) {
    TRY( VecDestroy(&qp->lambda_lb) );
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPComputeObjective"
/*@
   QPComputeObjective - Computes the objective.

   Collective on QP

   Input Parameter:
+  qp   - the QP
-  x    - the state vector

   Output Parameter:
.  f    - the objective value
   
   Notes:
   Computes -x^T(b - \frac{1}{2} Ax)

.seealso: QPComputeObjectiveGradient(), QPComputeObjectiveFromGradient(), QPComputeObjectiveFromGradient()
@*/
PetscErrorCode QPComputeObjective(QP qp, Vec x, PetscReal *f)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp,QP_CLASSID,1);
  PetscValidHeaderSpecific(x,VEC_CLASSID,2);
  PetscValidRealPointer(f,3);
  if (!qp->setupcalled) FLLOP_SETERRQ(PetscObjectComm((PetscObject)qp),PETSC_ERR_ORDER,"QPSetUp must be called first.");
  TRY( MatMult(qp->A,x,qp->xwork) );
  TRY( VecAYPX(qp->xwork,-0.5,qp->b) );
  TRY( VecDot(x,qp->xwork,f) );
  *f = -*f;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPComputeObjectiveGradient"
/*@
   QPComputeObjectiveGradient - Computes the gradient.

   Collective on QP

   Input Parameter:
+  qp   - the QP
-  x    - the state vector

   Output Parameter:
.  g    - the gradient value
   
   Notes:
   Computes Ax - b
   
.seealso: QPComputeObjective(), QPComputeObjectiveFromGradient(), QPComputeObjectiveAndGradient()
@*/
PetscErrorCode QPComputeObjectiveGradient(QP qp, Vec x, Vec g)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp,QP_CLASSID,1);
  PetscValidHeaderSpecific(x,VEC_CLASSID,2);
  PetscValidHeaderSpecific(g,VEC_CLASSID,3);
  if (!qp->setupcalled) FLLOP_SETERRQ(PetscObjectComm((PetscObject)qp),PETSC_ERR_ORDER,"QPSetUp must be called first.");
  TRY( MatMult(qp->A,x,g) );
  TRY( VecAXPY(g,-1.0,qp->b) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPComputeObjectiveFromGradient"
/*@
   QPComputeObjectiveFromGradient - Computes the objective.

   Collective on QP

   Input Parameter:
+  qp   - the QP
.  x    - the state vector
-  g    - the gradient

   Output Parameter:
.  f    - the objective value
   
   Notes:
   Computes x^T(g - b)/2
   
.seealso: QPComputeObjective(), QPComputeObjectiveGradient(), QPComputeObjectiveAndGradient()
@*/
PetscErrorCode QPComputeObjectiveFromGradient(QP qp, Vec x, Vec g, PetscReal *f)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp,QP_CLASSID,1);
  PetscValidHeaderSpecific(x,VEC_CLASSID,2);
  PetscValidRealPointer(f,4);
  PetscValidHeaderSpecific(g,VEC_CLASSID,3);
  if (!qp->setupcalled) FLLOP_SETERRQ(PetscObjectComm((PetscObject)qp),PETSC_ERR_ORDER,"QPSetUp must be called first.");

  TRY( VecWAXPY(qp->xwork,-1.0,qp->b,g) );
  TRY( VecDot(x,qp->xwork,f) );
  *f /= 2.0;
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "QPComputeObjectiveAndGradient"
/*@
   QPComputeObjective - Computes the objective and gradient.

   Collective on QP

   Input Parameter:
+  qp   - the QP
-  x    - the state vector

   Output Parameter:
+  g    - the gradient
-  f    - the objective value
   
   Notes:
   Computes -x^T(b - \frac{1}{2} Ax)

.seealso: QPComputeObjective(), QPComputeObjectiveGradient(), QPComputeObjectiveFromGradient()
@*/
PetscErrorCode QPComputeObjectiveAndGradient(QP qp, Vec x, Vec g, PetscReal *f)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp,QP_CLASSID,1);
  PetscValidHeaderSpecific(x,VEC_CLASSID,2);
  if (f) PetscValidRealPointer(f,4);
  if (g) PetscValidHeaderSpecific(g,VEC_CLASSID,3);
  if (!qp->setupcalled) FLLOP_SETERRQ(PetscObjectComm((PetscObject)qp),PETSC_ERR_ORDER,"QPSetUp must be called first.");

  if (!g) {
    TRY( QPComputeObjective(qp,x,f) );
    PetscFunctionReturn(0);
  }

  TRY( QPComputeObjectiveGradient(qp,x,g) );
  if (!f) PetscFunctionReturn(0);

  TRY( QPComputeObjectiveFromGradient(qp,x,g,f) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPPostSolve"
/*@
   QPPostSolve - Apply post solve function and optionaly view 

   Input Parameter:
.  qp   - the QP
  
   Options Database Key:
+  -qp_view            - view information about QP
.  -qp_chain_view      - view information about all QPs in the chain
.  -qp_chain_view_qppf - view information about all QPPFs in the chain
-  -qp_chain_view_kkt  - view how well are satisfied KKT conditions for each QP in the chain 
@*/
PetscErrorCode QPPostSolve(QP qp)
{
  PetscErrorCode (*postSolve)(QP,QP);
  QP parent, cqp;
  PetscBool flg, solved, view, first;
  PetscViewer v=NULL;
  PetscViewerFormat format;
  MPI_Comm comm;
  const char *prefix;

  PetscFunctionBeginI;
  PetscValidHeaderSpecific(qp,QP_CLASSID,1);
  TRY( PetscObjectGetComm((PetscObject)qp,&comm) );
  TRY( PetscObjectGetOptionsPrefix((PetscObject)qp,&prefix) );

  TRY( PetscOptionsGetViewer(comm,prefix,"-qp_view",&v,&format,&view) );
  if (view & !PetscPreLoadingOn) {
    TRY( PetscViewerPushFormat(v,format) );
    TRY( QPView(qp,v) );
    TRY( PetscViewerPopFormat(v) );
    TRY( PetscViewerDestroy(&v) );
  }

  TRY( PetscOptionsGetViewer(comm,prefix,"-qp_chain_view",&v,&format,&view) );
  if (view & !PetscPreLoadingOn) {
    TRY( PetscViewerPushFormat(v,format) );
    TRY( QPChainView(qp,v) );
    TRY( PetscViewerPopFormat(v) );
    TRY( PetscViewerDestroy(&v) );
  }

  TRY( PetscOptionsGetViewer(comm,prefix,"-qp_chain_view_qppf",&v,&format,&view) );
  if (view & !PetscPreLoadingOn) {
    TRY( PetscViewerPushFormat(v,format) );
    TRY( QPChainViewQPPF(qp,v) );
    TRY( PetscViewerPopFormat(v) );
    TRY( PetscViewerDestroy(&v) );
  }

  TRY( PetscOptionsGetViewer(comm,prefix,"-qp_chain_view_kkt",&v,&format,&view) );
  view &= !PetscPreLoadingOn;
  if (view) {
    TRY( PetscObjectTypeCompare((PetscObject)v,PETSCVIEWERASCII,&flg) );
    if (!flg) FLLOP_SETERRQ1(comm,PETSC_ERR_SUP,"Viewer type %s not supported",((PetscObject)v)->type_name);
    TRY( PetscViewerASCIIPrintf(v,"=====================\n") );
  }

  TRY( QPChainGetLast(qp,&cqp) );
  solved = cqp->solved;
  first = PETSC_TRUE;
  while (1) {
    TRY( QPRemoveInactiveBounds(cqp) );
    TRY( QPComputeMissingBoxMultipliers(cqp) );
    TRY( QPComputeMissingEqMultiplier(cqp) );
    parent = cqp->parent;
    postSolve = cqp->postSolve;
    if (postSolve) TRY( (*postSolve)(cqp,parent) );

    if (view) {
      if (first) {
        first = PETSC_FALSE;
      } else {
        TRY( PetscViewerASCIIPrintf(v, "-------------------\n") );
      }
      TRY( QPViewKKT(cqp,v) );
    }

    if (!parent) break;
    parent->solved = solved;
    if (cqp == qp) break;
    cqp = parent;
  }

  if (view) {
    TRY( PetscViewerASCIIPrintf(v,"=====================\n") );
    TRY( PetscViewerDestroy(&v) );
  }
  PetscFunctionReturnI(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPDestroy"
/*@
   QPDestroy - Destroys QP context.

   Collective on QP

   Input Parameter:
.  qp - QP context
@*/
PetscErrorCode QPDestroy(QP *qp)
{
  PetscFunctionBegin;
  if (!*qp) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(*qp,QP_CLASSID,1);
  if (--((PetscObject)(*qp))->refct > 0) {
    *qp = 0;
    PetscFunctionReturn(0);
  }
  if ((*qp)->postSolveCtxDestroy) TRY( (*qp)->postSolveCtxDestroy((*qp)->postSolveCtx) );
  TRY( QPReset(*qp) );
  TRY( QPPFDestroy(&(*qp)->pf) );
  TRY( PetscHeaderDestroy(qp) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPSetOperator"
/*@
   QPSetOperator - Sets the QP matrix and a symmetry flag.

   Input Parameter:
+  qp  - the QP
-  A   - the Hessian matrix
@*/
PetscErrorCode QPSetOperator(QP qp, Mat A)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp,QP_CLASSID,1);
  PetscValidHeaderSpecific(A,MAT_CLASSID,2);
  PetscCheckSameComm(qp,1,A,2);
  TRY( PCSetOperators(qp->pc,A,A) );
  if (A == qp->A) PetscFunctionReturn(0);

  TRY( MatDestroy(&qp->A) );
  qp->A = A;
  TRY( PetscObjectReference((PetscObject)A) );

  if (qp->changeListener) TRY( (*qp->changeListener)(qp) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPSetPC"
/*@
   QPSetPC - Sets preconditioner context.

   Input Parameter:
+  qp - the QP
-  pc - the preconditioner context
@*/
PetscErrorCode QPSetPC(QP qp, PC pc)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp,QP_CLASSID,1);
  PetscValidHeaderSpecific(pc,PC_CLASSID,2);
  PetscCheckSameComm(qp,1,pc,2);
  if (pc == qp->pc) PetscFunctionReturn(0);
  TRY( PCDestroy(&qp->pc) );
  qp->pc = pc;
  TRY( PetscObjectReference((PetscObject)pc) );
  TRY( PetscLogObjectParent((PetscObject)qp,(PetscObject)qp->pc) );
  if (qp->changeListener) TRY( (*qp->changeListener)(qp) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPGetOperator"
/*@
   QPGetOperator - Get the QP matrix.

   Input Parameter:
.  qp  - the QP

   Output Parameter:
.  A   - the matrix
@*/
PetscErrorCode QPGetOperator(QP qp,Mat *A)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp,QP_CLASSID,1);
  PetscValidPointer(A,2);
  *A = qp->A;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPGetPC"
/*@
   QPGetPC - Get preconditioner context.

   Input Parameter:
.  qp - the QP

   Output Parameter:
.  pc - the preconditioner context
@*/
PetscErrorCode QPGetPC(QP qp,PC *pc)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp,QP_CLASSID,1);
  PetscValidPointer(pc,2);
  if (!qp->pc) {
    TRY( PCCreate(PetscObjectComm((PetscObject)qp),&qp->pc) );
    TRY( PetscObjectIncrementTabLevel((PetscObject)qp->pc,(PetscObject)qp,0) );
    TRY( PetscLogObjectParent((PetscObject)qp,(PetscObject)qp->pc) );
    TRY( PCSetType(qp->pc,PCNONE) );
    TRY( PCSetOperators(qp->pc,qp->A,qp->A) );
  }
  *pc = qp->pc;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPSetOperatorNullSpace"
/*@
   QPSetOperatorNullSpace - Sets matrix with columns representing the null space of the QP operator.

   Input Parameter:
+  qp - the QP
-  R - null space matrix
@*/
PetscErrorCode QPSetOperatorNullSpace(QP qp,Mat R)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp,QP_CLASSID,1);
  if (R == qp->R) PetscFunctionReturn(0);
  TRY( MatDestroy(&qp->R) );
  qp->R = R;
  TRY( PetscObjectReference((PetscObject)R) );
  if (qp->changeListener) TRY( (*qp->changeListener)(qp) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPGetOperatorNullSpace"
/*@
   QPGetOperatorNullSpace - Get matrix with columns representing the null space of the QP operator.

   Input Parameter:
.  qp - the QP

   Output Parameter:
.  R - null space matrix
@*/
PetscErrorCode QPGetOperatorNullSpace(QP qp,Mat *R)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp,QP_CLASSID,1);
  PetscValidPointer(R,2);
  *R = qp->R;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPSetRhs"
/*@
   QPSetRhs - Set the QPs right hand side.

   Input Parameter:
+  qp - the QP
-  b - right hand side
@*/
PetscErrorCode QPSetRhs(QP qp,Vec b)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp,QP_CLASSID,1);
  PetscValidHeaderSpecific(b,VEC_CLASSID,2);
  PetscCheckSameComm(qp,1,b,2);
  if (b == qp->b) PetscFunctionReturn(0);

  TRY( VecDestroy(&qp->b) );
  qp->b = b;
  TRY( PetscObjectReference((PetscObject)b) );

  if (FllopDebugEnabled) {
    PetscReal norm;
    TRY( VecNorm(b,NORM_2,&norm) );
    TRY( FllopDebug1("||b|| = %0.2e\n", norm) );
  }

  if (qp->changeListener) TRY( (*qp->changeListener)(qp) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPGetRhs"
/*@
   QPGetRhs - Get the QPs right hand side.

   Input Parameter:
.  qp - the QP

   Output Parameter:
.  b - right hand side
@*/
PetscErrorCode QPGetRhs(QP qp,Vec *b)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp,QP_CLASSID,1);
  PetscValidPointer(b,2);
  *b = qp->b;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPSetIneq"
/*@
   QPSetIneq - Sets the inequality constraints.

   Input Parameter:
+  qp - the QP
.  Bineq - boolean matrix representing the inequality constraints placement 
-  cineq - vector prescribing inequality constraints
@*/
PetscErrorCode QPSetIneq(QP qp, Mat Bineq, Vec cineq)
{
  PetscReal norm;
  PetscBool change = PETSC_FALSE;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp,QP_CLASSID,1);

  if (Bineq) {
    PetscValidHeaderSpecific(Bineq,MAT_CLASSID,2);
    PetscCheckSameComm(qp,1,Bineq,2);
  }

  if (Bineq != qp->BI) {
    if (Bineq) TRY( PetscObjectReference((PetscObject)Bineq) );
    TRY( MatDestroy(&qp->BI) );
    TRY( MatDestroy(&qp->B) );
    TRY( QPSetIneqMultiplier(qp,NULL) );
    qp->BI = Bineq;
    change = PETSC_TRUE;
  }

  if (cineq) {
    if (!Bineq) {
      cineq = NULL;
      TRY( PetscInfo(qp, "null inequality constraint matrix specified, the constraint RHS vector will be ignored\n") );
    } else {
      PetscValidHeaderSpecific(cineq,VEC_CLASSID,3);
      PetscCheckSameComm(qp,1,cineq,3);
      TRY( VecNorm(cineq,NORM_2,&norm) );
      TRY( FllopDebug1("||cineq|| = %0.2e\n", norm) );
      if (norm < PETSC_MACHINE_EPSILON) {
        TRY( PetscInfo(qp, "zero inequality constraint RHS vector detected\n") );
        cineq = NULL;
      }
    }
  } else if (Bineq) {
    TRY( PetscInfo(qp, "null inequality constraint RHS vector handled as zero vector\n") );
  }

  if (cineq != qp->cI) {
    if (cineq) TRY( PetscObjectReference((PetscObject)cineq) );
    TRY( VecDestroy(&qp->cI) );
    TRY( VecDestroy(&qp->c) );
    qp->cI = cineq;
    change = PETSC_TRUE;
  }

  if (!Bineq) {
    TRY( VecDestroy(&qp->lambda_I) );
  }

  if (change) {
    if (qp->changeListener) TRY( (*qp->changeListener)(qp) );
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPGetIneq"
/*@
   QPGetIneq - Get the inequality constraints.

   Input Parameter:
.  qp - the QP

   Output Parameter:
+  Bineq - boolean matrix representing the inequality constraints placement 
-  cineq - vector prescribing inequality constraints
@*/
PetscErrorCode QPGetIneq(QP qp, Mat *Bineq, Vec *cineq)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp,QP_CLASSID,1);
  if (Bineq) {
    PetscValidPointer(Bineq, 2);
    *Bineq = qp->BI;
  }
  if (cineq) {
    PetscValidPointer(cineq, 3);
    *cineq = qp->cI;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPSetEq"
/*@
   QPSetEq - Sets the equality constraints.

   Input Parameter:
+  qp - the QP
.  Beq - boolean matrix representing the equality constraints placement 
-  ceq - vector prescribing equality constraints
@*/
PetscErrorCode QPSetEq(QP qp, Mat Beq, Vec ceq)
{
  PetscReal norm;
  PetscBool change = PETSC_FALSE;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp,QP_CLASSID,1);

  if (Beq) {
    PetscValidHeaderSpecific(Beq,MAT_CLASSID,2);
    PetscCheckSameComm(qp,1,Beq,2);
  }

  if (Beq != qp->BE) {
    if (Beq) {
      if (!qp->pf) {
        TRY( QPPFCreate(((PetscObject)qp)->comm,&qp->pf) );
        TRY( PetscLogObjectParent((PetscObject)qp,(PetscObject)qp->pf) );
        TRY( PetscObjectIncrementTabLevel((PetscObject)qp->pf,(PetscObject)qp,1) );
      }
      TRY( QPPFSetG(qp->pf, Beq) );

      TRY( PetscObjectReference((PetscObject)Beq) );
    }
    TRY( MatDestroy(&qp->BE) );
    TRY( MatDestroy(&qp->B) );
    TRY( QPSetEqMultiplier(qp,NULL) );
    qp->BE = Beq;
    qp->BE_nest_count = 0;
    change = PETSC_TRUE;
  }

  if (ceq) {
    if (!Beq) {
      ceq = NULL;
      TRY( PetscInfo(qp, "null equality constraint matrix specified, the constraint RHS vector will be ignored\n") );
    } else {
      PetscValidHeaderSpecific(ceq,VEC_CLASSID,3);
      PetscCheckSameComm(qp,1,ceq,3);
      TRY( VecNorm(ceq,NORM_2,&norm) );
      TRY( FllopDebug1("||ceq|| = %0.2e\n", norm) );
      if (norm < PETSC_MACHINE_EPSILON) {
        TRY( PetscInfo(qp, "zero equality constraint RHS vector detected\n") );
        ceq = NULL;
      }
    }
  } else if (Beq) {
    TRY( PetscInfo(qp, "null equality constraint RHS vector handled as zero vector\n") );
  }

  if (ceq != qp->cE) {
    if (ceq) TRY( PetscObjectReference((PetscObject)ceq) );
    TRY( VecDestroy(&qp->cE) );
    TRY( VecDestroy(&qp->c) );
    qp->cE = ceq;
    change = PETSC_TRUE;
  }

  if (!Beq) {
    TRY( VecDestroy(&qp->lambda_E) );
  }

  if (change) {
    if (qp->changeListener) TRY( (*qp->changeListener)(qp) );
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "QPAddEq"
/*@
   QPAddEq - Add the equality constraints.

   Input Parameter:
+  qp - the QP
.  Beq - boolean matrix representing the equality constraints placement 
-  ceq - vector prescribing equality constraints
@*/
PetscErrorCode QPAddEq(QP qp, Mat Beq, Vec ceq)
{
  PetscReal norm;
  Mat *subBE;
  PetscInt M, i;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp, QP_CLASSID, 1);
  PetscValidHeaderSpecific(Beq, MAT_CLASSID, 2);
  PetscCheckSameComm(qp, 1, Beq, 2);

  /* handle BE that has been set by QPSetEq */
  if (qp->BE && !qp->BE_nest_count) {
    Mat BE_orig=qp->BE;
    Vec cE_orig=qp->cE;

    TRY( PetscObjectReference((PetscObject)BE_orig) );
    if (cE_orig) TRY( PetscObjectReference((PetscObject)cE_orig) );
    TRY( QPSetEq(qp,NULL,NULL) );
    TRY( QPAddEq(qp,BE_orig,cE_orig) );
    TRY( MatDestroy(&BE_orig) );
    TRY( VecDestroy(&cE_orig) );
    FLLOP_ASSERT(qp->BE_nest_count==1,"qp->BE_nest_count==1");
  }
  
  M = qp->BE_nest_count++;

  TRY( PetscMalloc((M+1)*sizeof(Mat), &subBE) );   //Mat subBE[M+1];
  for (i = 0; i<M; i++) {
    TRY( MatNestGetSubMat(qp->BE, i, 0, &subBE[i]) );
    TRY( PetscObjectReference((PetscObject) subBE[i]) );
  }
  subBE[M] = Beq;

  TRY( MatDestroy(&qp->BE) );
  TRY( MatCreateNestPermon(PetscObjectComm((PetscObject) qp), M+1, NULL, 1, NULL, subBE, &qp->BE) );
  TRY( PetscObjectSetName((PetscObject)qp->BE,"BE") );

  if (!qp->pf) {
    TRY( QPPFCreate(((PetscObject) qp)->comm, &qp->pf) );
    TRY( PetscLogObjectParent((PetscObject)qp,(PetscObject)qp->pf) );
    TRY( PetscObjectIncrementTabLevel((PetscObject) qp->pf, (PetscObject) qp, 1) );
  }
  TRY( QPPFSetG(qp->pf, qp->BE) );

  if (ceq) {
    PetscValidHeaderSpecific(ceq, VEC_CLASSID, 3);
    PetscCheckSameComm(qp, 1, ceq, 3);
    TRY( VecNorm(ceq, NORM_2, &norm) );
    TRY( FllopDebug1("||ceq|| = %0.2e\n", norm) );
    if (norm < PETSC_MACHINE_EPSILON) {
      TRY( PetscInfo(qp, "zero equality constraint RHS vector detected\n") );
      ceq = NULL;
    }
  } else {
    TRY( PetscInfo(qp, "null equality constraint RHS vector handled as zero vector\n") );
  }

  if (ceq || qp->cE) {
    Vec *subCE;
    TRY( PetscMalloc((M+1)*sizeof(Vec), &subCE) );
    if (qp->cE) {
      for (i = 0; i<M; i++) {
        TRY( VecNestGetSubVec(qp->cE, i, &subCE[i]) );
        TRY( PetscObjectReference((PetscObject) subCE[i]) );
      }
      TRY( VecDestroy(&qp->cE) );
    } else {
      for (i = 0; i<M; i++) {
        TRY( MatCreateVecs(subBE[i], NULL, &subCE[i]) );
        TRY( VecSet(subCE[i], 0.0) );
      }
    }
    if (!ceq) {
      TRY( MatCreateVecs(Beq, NULL, &ceq) );
      TRY( VecSet(ceq, 0.0) );
    }
    subCE[M] = ceq;

    TRY( VecCreateNest(PetscObjectComm((PetscObject) qp), M+1, NULL, subCE, &qp->cE) );
    for (i = 0; i<M; i++) {
      TRY( VecDestroy( &subCE[i]) );
    }
    TRY( PetscFree(subCE) );
  }

  for (i = 0; i<M; i++) {
    TRY( MatDestroy(   &subBE[i]) );
  }
  TRY( PetscFree(subBE) );
  if (qp->changeListener) TRY( (*qp->changeListener)(qp) );

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPGetEqMultiplicityScaling"
PetscErrorCode QPGetEqMultiplicityScaling(QP qp, Vec *dE_new, Vec *dI_new)
{
  MPI_Comm comm;
  PetscInt i,ilo,ihi,j,k,ncols;
  PetscScalar multiplicity;
  Mat Bc=NULL, Bd=NULL, Bg=NULL;
  Mat Bct=NULL, Bdt=NULL, Bgt=NULL;
  PetscBool flg, scale_Bd=PETSC_TRUE, scale_Bc=PETSC_TRUE, count_Bd=PETSC_TRUE, count_Bc=PETSC_TRUE;
  Vec dof_multiplicities=NULL, edge_multiplicities_g=NULL, edge_multiplicities_d=NULL, edge_multiplicities_c=NULL;
  const PetscInt *cols;
  const PetscScalar *vals;
  
  PetscFunctionBeginI;
  TRY( PetscObjectGetComm((PetscObject)qp,&comm) );
  //TODO we now assume fully redundant case
  if (!qp->BE_nest_count) {
    Bg = qp->BE;
  } else {
    TRY( MatNestGetSubMat(qp->BE,0,0,&Bg) );
    if (qp->BE_nest_count >= 2) {
      TRY( MatNestGetSubMat(qp->BE,1,0,&Bd) );
    }
  }
  Bc = qp->BI;
  FLLOP_ASSERT(Bg,"Bg");
  
  if (!Bc) { scale_Bc = PETSC_FALSE; count_Bc = PETSC_FALSE; }
  if (!Bd) { scale_Bd = PETSC_FALSE; count_Bd = PETSC_FALSE; }
  TRY( PetscOptionsGetBool(NULL,NULL,"-qp_E_scale_Bd",&scale_Bd,NULL) );
  TRY( PetscOptionsGetBool(NULL,NULL,"-qp_E_scale_Bc",&scale_Bc,NULL) );
  TRY( PetscOptionsGetBool(NULL,NULL,"-qp_E_count_Bd",&count_Bd,NULL) );
  TRY( PetscOptionsGetBool(NULL,NULL,"-qp_E_count_Bc",&count_Bc,NULL) );
  //if (scale_Bc && !count_Bc) FLLOP_SETERRQ(PetscObjectComm((PetscObject)qp),PETSC_ERR_ARG_INCOMP,"-qp_E_scale_Bc implies -qp_E_count_Bc");
  //if (scale_Bd && !count_Bd) FLLOP_SETERRQ(PetscObjectComm((PetscObject)qp),PETSC_ERR_ARG_INCOMP,"-qp_E_scale_Bd implies -qp_E_count_Bd");

  TRY( MatGetOwnershipRangeColumn(Bg,&ilo,&ihi) );

  {
    TRY( MatCreateVecs(Bg,&dof_multiplicities,&edge_multiplicities_g) );
    TRY( VecSet(dof_multiplicities,   1.0) );
  }
  if (scale_Bd) {
    TRY( MatCreateVecs(Bd,NULL,&edge_multiplicities_d) );
    TRY( VecSet(edge_multiplicities_d,1.0) );
  }
  if (scale_Bc) {
    TRY( MatCreateVecs(Bc,NULL,&edge_multiplicities_c) );
    TRY( VecSet(edge_multiplicities_c,1.0) );
  }

  {
    TRY( MatIsImplicitTranspose(Bg,&flg) );
    FLLOP_ASSERT(flg,"Bg must be implicit transpose");
    TRY( FllopMatTranspose(Bg,MAT_TRANSPOSE_EXPLICIT,&Bgt) );
    for (i=ilo; i<ihi; i++) {
      TRY( MatGetRow(Bgt,i,&ncols,&cols,&vals) );
      k=0;
      for (j=0; j<ncols; j++) {
        if (vals[j]) k++;
      }
      TRY( MatRestoreRow(Bgt,i,&ncols,&cols,&vals) );
      if (k) {
        multiplicity = k+1;
        TRY( VecSetValue(dof_multiplicities,i,multiplicity,INSERT_VALUES) );
      }
    }
  }

  if (count_Bd) {
    TRY( MatIsImplicitTranspose(Bd,&flg) );
    FLLOP_ASSERT(flg,"Bd must be implicit transpose");
    TRY( FllopMatTranspose(Bd,MAT_TRANSPOSE_EXPLICIT,&Bdt) );
    for (i=ilo; i<ihi; i++) {
      TRY( MatGetRow(Bdt,i,&ncols,&cols,&vals) );
      k=0;
      for (j=0; j<ncols; j++) {
        if (vals[j]) k++;
        if (k>1) FLLOP_SETERRQ1(comm,PETSC_ERR_PLIB,"more than one nonzero in Bd row %d",i);
      }
      TRY( MatRestoreRow(Bdt,i,&ncols,&cols,&vals) );
      if (k) {
        TRY( VecGetValues(dof_multiplicities,1,&i,&multiplicity) );
        multiplicity++;
        TRY( VecSetValue(dof_multiplicities,i,multiplicity,INSERT_VALUES) );
      }
    }
  }

  if (count_Bc) {
    TRY( MatIsImplicitTranspose(Bc,&flg) );
    FLLOP_ASSERT(flg,"Bc must be implicit transpose");
    TRY( FllopMatTranspose(Bc,MAT_TRANSPOSE_EXPLICIT,&Bct) );
    for (i=ilo; i<ihi; i++) {
      TRY( MatGetRow(Bct,i,&ncols,&cols,&vals) );
      k=0;
      for (j=0; j<ncols; j++) {
        if (vals[j]) k++;
      }
      TRY( MatRestoreRow(Bct,i,&ncols,&cols,&vals) );
      if (k>1) TRY( PetscPrintf(comm,"WARNING: more than one nonzero in Bc row %d\n",i) );
      if (k) {
        TRY( VecGetValues(dof_multiplicities,1,&i,&multiplicity) );
        multiplicity++;
        TRY( VecSetValue(dof_multiplicities,i,multiplicity,INSERT_VALUES) );
      }
    }
  }

  TRY( VecAssemblyBegin(dof_multiplicities) );
  TRY( VecAssemblyEnd(dof_multiplicities) );
  TRY( VecSqrtAbs(dof_multiplicities) );
  TRY( VecReciprocal(dof_multiplicities) );

  {
    for (i=ilo; i<ihi; i++) {
      TRY( MatGetRow(Bgt,i,&ncols,&cols,NULL) );
      if (ncols) {
        TRY( VecGetValues(dof_multiplicities,1,&i,&multiplicity) );
        for (j=0; j<ncols; j++) {
          TRY( VecSetValue(edge_multiplicities_g,cols[j],multiplicity,INSERT_VALUES) );
        }
      }
      TRY( MatRestoreRow(Bgt,i,&ncols,&cols,NULL) );
    }
    TRY( VecAssemblyBegin(edge_multiplicities_g) );
    TRY( VecAssemblyEnd(  edge_multiplicities_g) );
    TRY( MatDestroy(&Bgt) );
  }

  if (scale_Bd) {
    TRY( FllopMatTranspose(Bd,MAT_TRANSPOSE_EXPLICIT,&Bdt) );
    for (i=ilo; i<ihi; i++) {
      TRY( MatGetRow(Bdt,i,&ncols,&cols,NULL) );
      if (ncols) {
        TRY( VecGetValues(dof_multiplicities,1,&i,&multiplicity) );
        for (j=0; j<ncols; j++) {
          TRY( VecSetValue(edge_multiplicities_d,cols[j],multiplicity,INSERT_VALUES) );
        }
      }
      TRY( MatRestoreRow(Bdt,i,&ncols,&cols,NULL) );
    }
    TRY( VecAssemblyBegin(edge_multiplicities_d) );
    TRY( VecAssemblyEnd(  edge_multiplicities_d) );
    TRY( MatDestroy(&Bdt) );
  }

  if (scale_Bc) {
    TRY( FllopMatTranspose(Bc,MAT_TRANSPOSE_EXPLICIT,&Bct) );
    for (i=ilo; i<ihi; i++) {
      TRY( MatGetRow(Bct,i,&ncols,&cols,NULL) );
      if (ncols) {
        TRY( VecGetValues(dof_multiplicities,1,&i,&multiplicity) );
        for (j=0; j<ncols; j++) {
          TRY( VecSetValue(edge_multiplicities_c,cols[j],multiplicity,INSERT_VALUES) );
        }
      }
      TRY( MatRestoreRow(Bct,i,&ncols,&cols,NULL) );
    }
    TRY( VecAssemblyBegin(edge_multiplicities_c) );
    TRY( VecAssemblyEnd(  edge_multiplicities_c) );
    TRY( MatDestroy(&Bct) );
  }

  if (edge_multiplicities_d) {
    Vec dE_vecs[2]={edge_multiplicities_g,edge_multiplicities_d};
    TRY( VecCreateNest(PetscObjectComm((PetscObject)qp),2,NULL,dE_vecs,dE_new) );
    TRY( VecDestroy(&edge_multiplicities_d) );
    TRY( VecDestroy(&edge_multiplicities_g) );
  } else {
    *dE_new = edge_multiplicities_g;
  }

  *dI_new = edge_multiplicities_c;

  TRY( VecDestroy(&dof_multiplicities) );
  PetscFunctionReturnI(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPGetEq"
/*@
   QPGetEq - Get the equality constraints.

   Input Parameter:
.  qp - the QP

   Output Parameter:
+  Beq - boolean matrix representing the equality constraints placement 
-  ceq - vector prescribing equality constraints
@*/
PetscErrorCode QPGetEq(QP qp, Mat *Beq, Vec *ceq)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp, QP_CLASSID, 1);
  if (Beq) {
    PetscValidPointer(Beq, 2);
    *Beq = qp->BE;
  }
  if (ceq) {
    PetscValidPointer(ceq, 3);
    *ceq = qp->cE;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPSetBox"
/*@
   QPSetBox - Sets the box constraints.

   Input Parameter:
+  qp - the QP
.  lb - lower bound
-  ub - upper bound
@*/
PetscErrorCode QPSetBox(QP qp, Vec lb, Vec ub)
{
  PetscReal extrem;
  MPI_Comm comm;
  PetscInt i;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp, QP_CLASSID, 1);
  if (lb) {
    PetscValidHeaderSpecific(lb,VEC_CLASSID,2);
    PetscCheckSameComm(qp,1,lb,2);
  }
  if (ub) {
    PetscValidHeaderSpecific(ub,VEC_CLASSID,3);
    PetscCheckSameComm(qp,1,ub,3);
  }
  if (lb == qp->lb && ub == qp->ub) PetscFunctionReturn(0);

  TRY( PetscObjectGetComm((PetscObject)qp,&comm) );

  if (lb) {
    TRY( VecMax(lb,&i,&extrem) );
    TRY( PetscInfo1(qp, "max(lb) = %0.2e\n", extrem) );
    if (extrem <= PETSC_NINFINITY) {
      TRY( PetscInfo1(qp, "inactive lower bound constraint detected\n", extrem) );
      lb = NULL;
    }
    else {
      TRY( PetscObjectReference((PetscObject) lb) );
    }
  } else {
    TRY( VecDestroy(&qp->lambda_lb) );
  }
  TRY( VecDestroy(&qp->lb) );
  qp->lb = lb;

  if (ub) {
    TRY( VecMin(ub,&i,&extrem) );
    TRY( PetscInfo1(qp, "min(ub) = %0.2e\n", extrem) );
    if (extrem >= PETSC_INFINITY) {
      TRY( PetscInfo1(qp, "inactive upper bound constraint detected\n", extrem) );
      ub = NULL;
    }
    else TRY( PetscObjectReference((PetscObject) ub) );
  } else {
    TRY( VecDestroy(&qp->lambda_ub) );
  }
  TRY( VecDestroy(&qp->ub) );
  qp->ub = ub;

  if (qp->changeListener) TRY( (*qp->changeListener)(qp) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPGetBox"
/*@
   QPGetBox - Get the box constraints.

   Input Parameter:
.  qp - the QP

   Output Parameter:
+  lb - lower bound
-  ub - upper bound
@*/
PetscErrorCode QPGetBox(QP qp, Vec *lb, Vec *ub)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp,QP_CLASSID,1);
  if (lb) {
    PetscValidPointer(lb, 2);
    *lb = qp->lb;
  }
  if (ub) {
    PetscValidPointer(ub, 3);
    *ub = qp->ub;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPSetEqMultiplier"
PetscErrorCode QPSetEqMultiplier(QP qp, Vec lambda_E)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp,QP_CLASSID,1);
  if (lambda_E == qp->lambda_E) PetscFunctionReturn(0);
  if (lambda_E) {
    PetscValidHeaderSpecific(lambda_E,VEC_CLASSID,2);
    PetscCheckSameComm(qp,1,lambda_E,2);
    TRY( PetscObjectReference((PetscObject)lambda_E) );
  }
  TRY( VecDestroy(&qp->lambda_E) );
  TRY( VecDestroy(&qp->lambda) );
  TRY( VecDestroy(&qp->Bt_lambda) );
  qp->lambda_E = lambda_E;
  if (qp->changeListener) TRY( (*qp->changeListener)(qp) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPSetIneqMultiplier"
PetscErrorCode QPSetIneqMultiplier(QP qp, Vec lambda_I)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp,QP_CLASSID,1);
  if (lambda_I == qp->lambda_I) PetscFunctionReturn(0);
  if (lambda_I) {
    PetscValidHeaderSpecific(lambda_I,VEC_CLASSID,2);
    PetscCheckSameComm(qp,1,lambda_I,2);
    TRY( PetscObjectReference((PetscObject)lambda_I) );
  }
  TRY( VecDestroy(&qp->lambda_I) );
  TRY( VecDestroy(&qp->lambda) );
  TRY( VecDestroy(&qp->Bt_lambda) );
  qp->lambda_I = lambda_I;
  if (qp->changeListener) TRY( (*qp->changeListener)(qp) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPSetLowerBoundMultiplier"
PetscErrorCode QPSetLowerBoundMultiplier(QP qp, Vec lambda_lb)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp,QP_CLASSID,1);
  if (lambda_lb == qp->lambda_lb) PetscFunctionReturn(0);
  if (lambda_lb) {
    PetscValidHeaderSpecific(lambda_lb,VEC_CLASSID,2);
    PetscCheckSameComm(qp,1,lambda_lb,2);
    TRY( PetscObjectReference((PetscObject)lambda_lb) );
  }
  TRY( VecDestroy(&qp->lambda_lb) );
  qp->lambda_lb = lambda_lb;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPSetUpperBoundMultiplier"
PetscErrorCode QPSetUpperBoundMultiplier(QP qp, Vec lambda_ub)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp,QP_CLASSID,1);
  if (lambda_ub == qp->lambda_ub) PetscFunctionReturn(0);
  if (lambda_ub) {
    PetscValidHeaderSpecific(lambda_ub,VEC_CLASSID,2);
    PetscCheckSameComm(qp,1,lambda_ub,2);
    TRY( PetscObjectReference((PetscObject)lambda_ub) );
  }
  TRY( VecDestroy(&qp->lambda_ub) );
  qp->lambda_ub = lambda_ub;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPSetInitialVector"
/*@
   QPSetInitialVector - Sets the inital guess.

   Input Parameter:
.  qp - the QP
-  x  - initial guess
@*/
PetscErrorCode QPSetInitialVector(QP qp,Vec x)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp,QP_CLASSID,1);
  if (x) {
    PetscValidHeaderSpecific(x,VEC_CLASSID,2);
    PetscCheckSameComm(qp,1,x,2);
  }

  TRY( VecDestroy(&qp->x) );
  qp->x = x;

  if (x) {
    TRY( PetscObjectReference((PetscObject)x) );
    if (FllopDebugEnabled) {
      PetscReal norm;
      TRY( VecNorm(x,NORM_2,&norm) );
      TRY( FllopDebug1("||x|| = %0.2e\n", norm) );
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPGetSolutionVector"
/*@
   QPGetSolutionVector - Get solution vector.

   Input Parameter:
.  qp - the QP

   Output Parameter:
.  x - solution vector
@*/
PetscErrorCode QPGetSolutionVector(QP qp,Vec *x)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp,QP_CLASSID,1);
  PetscValidPointer(x,2);
  TRY( QPInitializeInitialVector_Private(qp) );
  *x = qp->x;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPSetWorkVector"
/*@
   QPSetWorkVector - Set work vector.

   Input Parameter:
+  qp    - the QP
-  xwork - work vector
@*/
PetscErrorCode QPSetWorkVector(QP qp,Vec xwork)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp,QP_CLASSID,1);
  if (xwork == qp->xwork) PetscFunctionReturn(0);
  if (xwork) {
    PetscValidHeaderSpecific(xwork,VEC_CLASSID,2);
    PetscCheckSameComm(qp,1,xwork,2);
    TRY( PetscObjectReference((PetscObject)xwork) );
  }
  TRY( VecDestroy(&qp->xwork) );
  qp->xwork = xwork;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPGetVecs"
/*@
   QPGetVecs - Get vector(s) compatible with the QP operator matrix, i.e. with the same
     parallel layout

   Input Parameter:
.  qp - the QP

   Output Parameter:
+   right - (optional) vector that the matrix can be multiplied against
-   left - (optional) vector that the matrix vector product can be stored in

   Notes: These are new vectors which are not owned by the QP, they should be destroyed in VecDestroy() when no longer needed
@*/
PetscErrorCode QPGetVecs(QP qp,Vec *right,Vec *left)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp,QP_CLASSID,1);
  PetscValidType(qp,1);
  if (qp->A) {
    TRY( MatCreateVecs(qp->A,right,left) );
  } else {
    FLLOP_SETERRQ(((PetscObject)qp)->comm,PETSC_ERR_ORDER,"system operator not set yet");
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPSetChangeListener"
PetscErrorCode QPSetChangeListener(QP qp,PetscErrorCode (*f)(QP))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp,QP_CLASSID,1);
  qp->changeListener = f;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPGetChangeListener"
PetscErrorCode QPGetChangeListener(QP qp,PetscErrorCode (**f)(QP))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp,QP_CLASSID,1);
  *f = qp->changeListener;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPSetChangeListenerContext"
PetscErrorCode QPSetChangeListenerContext(QP qp,void *ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp,QP_CLASSID,1);
  qp->changeListenerCtx = ctx;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPGetChangeListenerContext"
PetscErrorCode QPGetChangeListenerContext(QP qp,void *ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp,QP_CLASSID,1);
  PetscValidPointer(ctx,2);
  *(void**)ctx = qp->changeListenerCtx;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPGetChild"
/*@
   QPGetChild - Get QP child.

   Input Parameter:
.  qp - the QP

   Output Parameter:
.  child - QP child
@*/
PetscErrorCode QPGetChild(QP qp,QP *child)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp,QP_CLASSID,1);
  PetscValidPointer(child,2);
  *child = qp->child;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPGetParent"
/*@
   QPGetParent - Get QP parent.

   Input Parameter:
.  qp - the QP

   Output Parameter:
.  parent - QP parent
@*/
PetscErrorCode QPGetParent(QP qp,QP *parent)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp,QP_CLASSID,1);
  PetscValidPointer(parent,2);
  *parent = qp->parent;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPGetPostSolve"
/*@
   QPGetPostSolve - Get QP post solve function.

   Input Parameter:
.  qp - the QP

   Output Parameter:
.  f - post solve function
@*/
PetscErrorCode QPGetPostSolve(QP qp, PetscErrorCode (**f)(QP,QP))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp,QP_CLASSID,1);
  *f = qp->postSolve;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPGetTransform"
/*@
   QPGetTransform - Get QP transform function.

   Input Parameter:
.  qp - the QP

   Output Parameter:
.  f - transform function
@*/
PetscErrorCode QPGetTransform(QP qp,PetscErrorCode(**f)(QP))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp,QP_CLASSID,1);
  *f = qp->transform;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPGetQPPF"
/*@
   QPGetQPPF - Get QPPF associated with QP.

   Input Parameter:
.  qp - the QP

   Output Parameter:
.  pf - the QPPF
@*/
PetscErrorCode QPGetQPPF(QP qp, QPPF *pf)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp,QP_CLASSID,1);
  PetscValidPointer(pf,2);
  *pf = qp->pf;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPSetQPPF"
/*@
   QPSetQPPF - Set QPPF into QP.

   Input Parameter:
+  qp - the QP
-  pf - the QPPF
@*/
PetscErrorCode QPSetQPPF(QP qp, QPPF pf)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp,QP_CLASSID,1);
  if (pf) {
    PetscValidHeaderSpecific(pf,QPPF_CLASSID,2);
    TRY( PetscObjectReference((PetscObject)pf) );
  }
  TRY( QPPFDestroy(&qp->pf) );
  qp->pf = pf;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPIsSolved"
/*@
   QPIsSolved - Flag is true if solved the QP is solved.

   Input Parameter:
.  qp  - the QP
   
   Output Parameter
.  flg - true if solved
@*/
PetscErrorCode QPIsSolved(QP qp,PetscBool *flg)
{
  PetscFunctionBegin;
  *flg = qp->solved;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "QPSetOptionsPrefix"
/*@
   QPSetOptionsPrefix - Sets the prefix used for searching for all
   QP options in the database.

   Input Parameters:
+  qp - the QP
-  prefix - the prefix string to prepend to all QP option requests

   Notes:
   A hyphen (-) must NOT be given at the beginning of the prefix name.
   The first character of all runtime options is AUTOMATICALLY the
   hyphen.

.seealso: QPAppendOptionsPrefix(), QPGetOptionsPrefix()
@*/
PetscErrorCode QPSetOptionsPrefix(QP qp,const char prefix[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp,QP_CLASSID,1);
  TRY( PetscObjectSetOptionsPrefix((PetscObject)qp,prefix) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "QPAppendOptionsPrefix"
/*@
   QPAppendOptionsPrefix - Appends to the prefix used for searching for all
   QP options in the database.

   Input Parameters:
+  QP - the QP
-  prefix - the prefix string to prepend to all QP option requests

   Notes:
   A hyphen (-) must NOT be given at the beginning of the prefix name.
   The first character of all runtime options is AUTOMATICALLY the hyphen.

.seealso: QPSetOptionsPrefix(), QPGetOptionsPrefix()
@*/
PetscErrorCode QPAppendOptionsPrefix(QP qp,const char prefix[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp,QP_CLASSID,1);
  TRY( PetscObjectAppendOptionsPrefix((PetscObject)qp,prefix) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "QPGetOptionsPrefix"
/*@
   QPGetOptionsPrefix - Gets the prefix used for searching for all
   QP options in the database.

   Input Parameters:
.  qp - the Krylov context

   Output Parameters:
.  prefix - pointer to the prefix string used is returned

.seealso: QPSetOptionsPrefix(), QPAppendOptionsPrefix()
@*/
PetscErrorCode QPGetOptionsPrefix(QP qp,const char *prefix[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp,QP_CLASSID,1);
  TRY( PetscObjectGetOptionsPrefix((PetscObject)qp,prefix) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPSetFromOptions_Private"
static PetscErrorCode QPSetFromOptions_Private(QP qp)
{
  PetscFunctionBegin;
  if (!qp->setfromoptionscalled) PetscFunctionReturn(0);

  _fllop_ierr = PetscObjectOptionsBegin((PetscObject)qp);CHKERRQ(_fllop_ierr);
  if (!qp->pc) TRY( QPGetPC(qp,&qp->pc) );

  if (qp->pf) {
    TRY( FllopPetscObjectInheritPrefixIfNotSet((PetscObject)qp->pf,(PetscObject)qp,NULL) );
  }
  TRY( FllopPetscObjectInheritPrefixIfNotSet((PetscObject)qp->pc,(PetscObject)qp,NULL) );

  if (qp->pf) {
    TRY( QPPFSetFromOptions(qp->pf) );
  }
  TRY( PCSetFromOptions(qp->pc) );

  _fllop_ierr = PetscOptionsEnd();CHKERRQ(_fllop_ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPSetFromOptions"
/*@
   QPSetFromOptions - Sets QP options from the options database.

   Collective on QP

   Input Parameters:
.  qp - the QP context

   Options Database Keys:
+  -qp_E_scale_type multiplicity - 
.	 -qp_E_remove_gluing_of_dirichlet - 
.  -qp_E_count_Bd - 
.  -qp_E_scale_Bd - 
.  -qp_E_count_Bc - 
.  -qp_E_scale_Bc - 
.  -qp_view            - view information about QP
.  -qp_chain_view      - view information about all QPs in the chain
.  -qp_chain_view_kkt  - view how well are satisfied KKT conditions for each QP in the chain 
-  -qp_chain_view_qppf - view information about all QPPFs in the chain

   Notes:
   To see all options, run your program with the -help option
   or consult Users-Manual: ch_qp
@*/
PetscErrorCode QPSetFromOptions(QP qp)
{
  PetscBool flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp,QP_CLASSID,1);
  _fllop_ierr = PetscObjectOptionsBegin((PetscObject)qp);CHKERRQ(_fllop_ierr);

  /* options processed elsewhere */
  TRY( PetscOptionsName("-qp_view","print the QP info at the end of a QPSSolve call","QPView",&flg) );

  /* process any options handlers added with PetscObjectAddOptionsHandler() */
  TRY( PetscObjectProcessOptionsHandlers(PetscOptionsObject,(PetscObject)qp) );
  _fllop_ierr = PetscOptionsEnd();CHKERRQ(_fllop_ierr);
  qp->setfromoptionscalled++;
  PetscFunctionReturn(0);
}
