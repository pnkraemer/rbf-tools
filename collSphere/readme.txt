
Summary:

* Classic tps collocation works well and behaves well with the preconditioner (this seems to be a performance baseline)
* Higher order TPS collocation has a worse condition number but seems to benefit from the preconditioner; the radius has to be improved significantly to achieve the same benefit (from 7 to ~15 or even more?)
* Unsymmetric collocation seems to behave similarly to the higher order tps collocation
* The condition numbers of interpolation and collocation are yet to be compared thoroughly, their required radii seem to be similar
* Symmetric collocation seems to benefit from the preconditioner as well, however, the small systems are ill-conditioned so it is not fun to test on large pointsets (it has only been checked for few points for which it already is incredibly slow)
* Symmetric coll. has not even been tested for even 500 points; for 150 points and a larg(ish) radius of 15 (which gives N = 71 points) the condition number reduced from 30000 to 100 which seems to be an indicator for a decent benefit
* All of these tests are not yet checked for large pointsets, thus decent radii are only conjectured and not shown
* Order-3 seems to outperform order-2 interpolation and collocation, at least for small(ish) pointsets
* Symmetric collocation seems to have no benefit over unsymmetric collocation (since symmetry gets destroyed anyway)


Again, but shorter:

* Classic version works well
* Higher order interpolation and unsymmetric collocation seems to benefit from preconditioning, although the radius as to be scaled up
* Order-3 symmetric collocation seems to benefit from preconditioning as well but is extremely slow already for small points and therefore not investigated well yet

Perhaps next up:
* Investigate good radii for the collocation method(s) - at which radii do we have a constant number of iterations in gmres?
* Check the larger pointsets (for all but the baseline programm)




Last changed: July 20, 2018

Nicholas Krämer
kraemer@ins.uni-bonn.de

