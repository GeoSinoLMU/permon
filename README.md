PermonQP - the PERMON core QP solver
================================

Project homepage: <http://permon.it4i.cz>  
GitHub: <https://github.com/It4innovations/permon>

Please use [GitHub](https://github.com/It4innovations/permon) for issues and pull requests.

Quick guide to PermonQP installation
------------------------------------

1. set environment variables `PETSC_DIR` and `PETSC_ARCH` pointing to a PETSc instance
   - `PETSC_ARCH` may is empty in case of "prefix" PETSc installation
   - for more details about [PETSc](http://www.mcs.anl.gov/petsc/) installation and the two environment variables, see [PETSc documentation](http://www.mcs.anl.gov/petsc/documentation/installation.html)
2. set `PERMON_DIR` variable pointing to the PermonQP directory (probably this file's parent directory)
3. build PermonQP simply using makefile (makes use of PETSc buildsystem):

     `make`
4. if the build is successful, there is a new subdirectory named `$PETSC_ARCH`, the program library is `$PETSC_ARCH/lib/libpermon.{so,a}`
   - shared library (.so) is built just if PETSc has been configured with option `--with-shared-libraries`
   - all compiler settings are inherited from PETSc

Currently supported PETSc versions
----------------------------------
* 3.6.\*
* 3.7.\*
