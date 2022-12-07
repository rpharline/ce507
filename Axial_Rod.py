# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 15:16:33 2022

@author: rphar
"""

import Basis_Func as basis
import unittest
import math
import sympy
import scipy
import numpy
import uspline
import bext
import Quadrature as quadrature
from matplotlib import pyplot as plt

def assembleStiffnessMatrix(continuity,problem, uspline_bext):
    E = problem["elastic_modulus"]
    A = problem["area"]
    num_elems = bext.getNumElems(uspline_bext)
    degree_list = []
    K_list = []
    
    for i in range(0,num_elems):
        degree = bext.getElementDegree(uspline_bext, i)
        degree_list.append(degree)
        n = int(numpy.ceil((2*degree+1)/2))
        qp,w = quadrature.computeGaussLegendreQuadrature(n)
        domain = bext.getElementDomain(uspline_bext, i)
        extraction_operator = bext.getElementExtractionOperator(uspline_bext, i)
        
        K = numpy.zeros((degree+1,degree+1))
        jacob = (basis.Jacobian([-1,1], domain))**(-1)
        for a in range(degree+1):
            for b in range(degree+1):
                for q in range(len(qp)):
                    N_A = basis.evalSplineBasisDeriv1D(extraction_operator, a, 1, [-1,1], qp[q])
                    N_B = basis.evalSplineBasisDeriv1D(extraction_operator, b, 1, [-1,1], qp[q])
                    K[a,b] += E * A * N_A * N_B * w[q] * jacob
                    
        K_list.append(K)
    K = local_to_globalK(K_list, degree_list, continuity)
    return K

def local_to_globalK(K_list, degree, continuity):
    del continuity[0]
    del continuity[-1]
    dim = sum(degree) - sum(continuity) + 1
    K = numpy.zeros((dim,dim))
    
    for i in range(len(K_list)):
        for a in range(0, degree[i]+1):
            if i == 0:
                A = i * degree[i] + a
            else:
                A = i * degree[i] + a - continuity[i-1]
            for b in range(degree[i]+1):
                if i == 0:
                    B = i * degree[i] + b
                else:
                    B = i * degree[i] + b - continuity[i-1]
                K[A,B] += K_list[i][a,b]
    
    return K

def assembleForceVector(continuity ,problem, uspline_bext ):
    target_fun = problem["body_force"]
    num_elems = bext.getNumElems(uspline_bext)
    degree_list = []
    F_list = []
    
    for i in range(0,num_elems):
        degree = bext.getElementDegree(uspline_bext, i)
        degree_list.append(degree)
        n = int(numpy.ceil((2*degree+1)/2))
        xi_qp, w_qp = quadrature.computeGaussLegendreQuadrature(n)
        domain = bext.getElementDomain(uspline_bext, i)
        extraction_operator = bext.getElementExtractionOperator(uspline_bext, i)
        F_term = numpy.zeros((degree+1))
        jacob = basis.Jacobian(original_domain = [-1,1], output_domain = domain)
        for A in range(0,degree+1):
            for q in range(0,len(xi_qp)):
                N_A = basis.evalSplineBasis1D(extraction_operator, A, [-1,1], xi_qp[q])
                
                F_term[A] += N_A * target_fun * w_qp[q] * jacob 
        F_list.append(F_term)
    F = Local_to_Global_F_Matrix(F_list,degree_list,continuity)
    return F
    
    force_vector = Local_to_Global_F_Matrix(F_list, degree_list, continuity)
    force_vector = applyTraction(problem, force_vector, uspline_bext)
    return force_vector

def Local_to_Global_F_Matrix(F_list,degree,continuity):
    del continuity[0]
    del continuity[-1]
    dim = sum(degree) - sum(continuity) + 1
    F = numpy.zeros((dim))
    for i in range(0,len(F_list)):
        for a in range(0,degree[i]+1):
            if i == 0:
                A = i * (degree[i]) + a
            else:
                A = i * (degree[i]) + a - continuity[i-1]
            F[A] += F_list[i][a]

    return F

def applyTraction( problem, force_vector, uspline_bext ):
    point = problem["traction"]["position"]
    traction = problem["traction"]["value"]
    elem_id = bext.getElementIdContainingPoint(uspline_bext, point)
    degree = bext.getElementDegree(uspline_bext, elem_id)
    extraction_operator = bext.getElementExtractionOperator(uspline_bext, elem_id)
    domain = bext.getElementDomain(uspline_bext, elem_id)
    for i in range(0,degree+1):
        A = bext.getElementNodeIds(uspline_bext, elem_id)
        force_vector[A] = basis.evalSplineBasis1D(extraction_operator, i, domain, point) * traction
    return force_vector

def applyDisplacement(force_vector, stiffness_matrix, problem, uspline_bext):
    displacement = problem["displacement"]["value"]
    position =  problem["displacement"]["value"]
    elem_id = bext.getElementIdContainingPoint(uspline_bext, position)
    degree = bext.getElementDegree(uspline_bext, elem_id)
    
    for i in range(0,degree+1):
        A = bext.getElementNodeIds(uspline_bext, elem_id)
        force_vector[A] += force_vector[i] - stiffness_matrix[i,0] * displacement
    
    force_vector = numpy.delete(force_vector,0)
    stiffness_matrix = numpy.delete(stiffness_matrix,0,0)
    stiffness_matrix = numpy.delete(stiffness_matrix,0,1)
    return stiffness_matrix,force_vector

def computeSolution(continuity,problem, uspline_bext):
    continuity2 = []
    for i in range(0,len(continuity)):
        continuity2.append(continuity[i])
    K = assembleStiffnessMatrix(continuity, problem, uspline_bext)
    F = assembleForceVector(continuity2, problem, uspline_bext)
    K,F = applyDisplacement(F, K, problem, uspline_bext)
    
    F = F.transpose()
    d = numpy.linalg.solve(K,F)
    d = numpy.insert(d,0,0)
    return d

def evaluateSolutionAt( x, coeff, uspline_bext ):
    elem_id = bext.getElementIdContainingPoint( uspline_bext, x )
    elem_nodes = bext.getElementNodeIds( uspline_bext, elem_id )
    elem_domain = bext.getElementDomain( uspline_bext, elem_id )
    elem_degree = bext.getElementDegree( uspline_bext, elem_id )
    elem_extraction_operator = bext.getElementExtractionOperator( uspline_bext, elem_id )
    sol = 0.0
    for n in range( 0, len( elem_nodes ) ):
        curr_node = elem_nodes[n]
        sol += coeff[curr_node] * basis.evalSplineBasis1D( extraction_operator = elem_extraction_operator, basis_idx = n, domain = elem_domain, variate = x )
    return sol

def computeElementFitError( problem, coeff, uspline_bext, elem_id ):
    domain = bext.getDomain( uspline_bext )
    elem_domain = bext.getElementDomain( uspline_bext, elem_id )
    elem_degree = bext.getElementDegree( uspline_bext, elem_id )
    num_qp = int( numpy.ceil( ( 2*(elem_degree - 1) + 1 ) / 2.0 ) + 1 )
    abs_err_fun = lambda x : abs( evaluateExactSolutionAt( problem, basis.affine_mapping_1D( [-1, 1], elem_domain, x ) ) - evaluateSolutionAt( basis.affine_mapping_1D( [-1, 1], elem_domain, x ), coeff, uspline_bext ) )
    abs_error = quadrature.quad( abs_err_fun, elem_domain, num_qp )
    return abs_error

def computeFitError( problem, coeff, uspline_bext ):
    num_elems = bext.getNumElems( uspline_bext )
    abs_error = 0.0
    for elem_idx in range( 0, num_elems ):
        elem_id = bext.elemIdFromElemIdx( uspline_bext, elem_idx )
        abs_error += computeElementFitError( problem, coeff, uspline_bext, elem_id )
    domain = bext.getDomain( uspline_bext )
    target_fun_norm, _ = scipy.integrate.quad( lambda x: abs( evaluateExactSolutionAt( problem, x ) ), domain[0], domain[1], epsrel = 1e-12, limit = num_elems * 100 )
    rel_error = abs_error / target_fun_norm
    return abs_error, rel_error

def plotCompareGoldTestSolution( gold_coeff, test_coeff, uspline_bext ):
    domain = bext.getDomain( uspline_bext )
    x = numpy.linspace( domain[0], domain[1], 1000 )
    yg = numpy.zeros( 1000 )
    yt = numpy.zeros( 1000 )
    for i in range(0, len(x) ):
        yg[i] = evaluateSolutionAt( x[i], test_coeff, uspline_bext )
        yt[i] = evaluateSolutionAt( x[i], gold_coeff, uspline_bext )
    plt.plot( x, yg )
    plt.plot( x, yt )
    plt.show()

def plotCompareFunToExactSolution( problem, test_coeff, uspline_bext ):
    domain = bext.getDomain( uspline_bext )
    x = numpy.linspace( domain[0], domain[1], 1000 )
    ya = numpy.zeros( 1000 )
    ye = numpy.zeros( 1000 )
    for i in range(0, len(x) ):
        ya[i] = evaluateSolutionAt( x[i], test_coeff, uspline_bext )
        ye[i] = evaluateExactSolutionAt( problem, x[i] )
    plt.plot( x, ya )
    plt.plot( x, ye )
    plt.show()

def computeConvergenceRate( num_entities, qoi ):
    def func( x, a, b, c ):
        return a * numpy.power( x, b ) + c
    fit = scipy.optimize.curve_fit(func, num_entities, qoi, method='trf', bounds = ([-numpy.inf, -numpy.inf, -numpy.inf ], [numpy.inf, 0.0, numpy.inf]) )
    a,b,c = fit[0]
    return b

def plotSolution( sol_coeff, uspline_bext ):
    domain = bext.getDomain( uspline_bext )
    x = numpy.linspace( domain[0], domain[1], 1000 )
    y = numpy.zeros( 1000 )
    for i in range(0, len(x) ):
        y[i] = evaluateSolutionAt( x[i], sol_coeff, uspline_bext )
    plt.plot( x, y )
    plt.plot( bext.getSplineNodes( uspline_bext )[:,0], sol_coeff, color = "k", marker = "o", markerfacecolor = "k" )
    plt.show()

def evaluateExactSolutionAt( problem, x ):
    term_1 = problem[ "traction" ][ "value" ] / evaluateConstitutiveModel( problem ) * x
    term_2 = problem[ "displacement" ][ "value" ]
    term_3 =  ( ( problem[ "length" ]**2.0 * problem[ "body_force" ] / 2 ) / evaluateConstitutiveModel( problem ) ) - ( ( ( problem[ "length" ] - x )**2.0 * problem[ "body_force" ] / 2 ) / evaluateConstitutiveModel( problem ) )
    sol = term_1 + term_2 + term_3
    return sol

def plotExactSolution( problem ):
    domain = [0, problem[ "length" ] ]
    x = numpy.linspace( domain[0], domain[1], 1000 )
    y = numpy.zeros( 1000 )
    for i in range(0, len(x) ):
        y[i] = evaluateExactSolutionAt( problem, x[i] )
    plt.plot( x, y )
    plt.show()

# class Test_evalBernsteinBasisDeriv( unittest.TestCase ):
#        def test_constant_at_nodes( self ):
#               self.assertAlmostEqual( first = basis.evalBernsteinBasisDeriv( degree = 0, basis_idx = 0, deriv = 0, domain = [0, 1], variate = 0.0 ), second = 1.0, delta = 1e-12 )
#               self.assertAlmostEqual( first = basis.evalBernsteinBasisDeriv( degree = 0, basis_idx = 0, deriv = 0, domain = [0, 1], variate = 1.0 ), second = 1.0, delta = 1e-12 )

#        def test_constant_1st_deriv_at_nodes( self ):
#               self.assertAlmostEqual( first = basis.evalBernsteinBasisDeriv( degree = 0, basis_idx = 0, deriv = 1, domain = [0, 1], variate = 0.0 ), second = 0.0, delta = 1e-12 )
#               self.assertAlmostEqual( first = basis.evalBernsteinBasisDeriv( degree = 0, basis_idx = 0, deriv = 1, domain = [0, 1], variate = 1.0 ), second = 0.0, delta = 1e-12 )

#        def test_constant_2nd_deriv_at_nodes( self ):
#               self.assertAlmostEqual( first = basis.evalBernsteinBasisDeriv( degree = 0, basis_idx = 0, deriv = 2, domain = [0, 1], variate = 0.0 ), second = 0.0, delta = 1e-12 )
#               self.assertAlmostEqual( first = basis.evalBernsteinBasisDeriv( degree = 0, basis_idx = 0, deriv = 2, domain = [0, 1], variate = 1.0 ), second = 0.0, delta = 1e-12 )

#        def test_linear_at_nodes( self ):
#               self.assertAlmostEqual( first = basis.evalBernsteinBasisDeriv( degree = 1, basis_idx = 0, deriv = 0, domain = [0, 1], variate = 0.0 ), second = 1.0, delta = 1e-12 )
#               self.assertAlmostEqual( first = basis.evalBernsteinBasisDeriv( degree = 1, basis_idx = 0, deriv = 0, domain = [0, 1], variate = 1.0 ), second = 0.0, delta = 1e-12 )
#               self.assertAlmostEqual( first = basis.evalBernsteinBasisDeriv( degree = 1, basis_idx = 1, deriv = 0, domain = [0, 1], variate = 0.0 ), second = 0.0, delta = 1e-12 )
#               self.assertAlmostEqual( first = basis.evalBernsteinBasisDeriv( degree = 1, basis_idx = 1, deriv = 0, domain = [0, 1], variate = 1.0 ), second = 1.0, delta = 1e-12 )

#        def test_linear_at_gauss_pts( self ):
#               self.assertAlmostEqual( first = basis.evalBernsteinBasisDeriv( degree = 1, basis_idx = 0, deriv = 0, domain = [0, 1], variate =  0.5 ), second = 0.5, delta = 1e-12 )
#               self.assertAlmostEqual( first = basis.evalBernsteinBasisDeriv( degree = 1, basis_idx = 1, deriv = 0, domain = [0, 1], variate =  0.5 ), second = 0.5, delta = 1e-12 )

#        def test_quadratic_at_nodes( self ):
#               self.assertAlmostEqual( first = basis.evalBernsteinBasisDeriv( degree = 2, basis_idx = 0, deriv = 0, domain = [0, 1], variate = 0.0 ), second = 1.00, delta = 1e-12 )
#               self.assertAlmostEqual( first = basis.evalBernsteinBasisDeriv( degree = 2, basis_idx = 0, deriv = 0, domain = [0, 1], variate = 0.5 ), second = 0.25, delta = 1e-12 )
#               self.assertAlmostEqual( first = basis.evalBernsteinBasisDeriv( degree = 2, basis_idx = 0, deriv = 0, domain = [0, 1], variate = 1.0 ), second = 0.00, delta = 1e-12 )
#               self.assertAlmostEqual( first = basis.evalBernsteinBasisDeriv( degree = 2, basis_idx = 1, deriv = 0, domain = [0, 1], variate = 0.0 ), second = 0.00, delta = 1e-12 )
#               self.assertAlmostEqual( first = basis.evalBernsteinBasisDeriv( degree = 2, basis_idx = 1, deriv = 0, domain = [0, 1], variate = 0.5 ), second = 0.50, delta = 1e-12 )
#               self.assertAlmostEqual( first = basis.evalBernsteinBasisDeriv( degree = 2, basis_idx = 1, deriv = 0, domain = [0, 1], variate = 1.0 ), second = 0.00, delta = 1e-12 )
#               self.assertAlmostEqual( first = basis.evalBernsteinBasisDeriv( degree = 2, basis_idx = 2, deriv = 0, domain = [0, 1], variate = 0.0 ), second = 0.00, delta = 1e-12 )
#               self.assertAlmostEqual( first = basis.evalBernsteinBasisDeriv( degree = 2, basis_idx = 2, deriv = 0, domain = [0, 1], variate = 0.5 ), second = 0.25, delta = 1e-12 )
#               self.assertAlmostEqual( first = basis.evalBernsteinBasisDeriv( degree = 2, basis_idx = 2, deriv = 0, domain = [0, 1], variate = 1.0 ), second = 1.00, delta = 1e-12 )

#        def test_quadratic_at_gauss_pts( self ):
#               x = [ -1.0 / math.sqrt(3.0) , 1.0 / math.sqrt(3.0) ]
#               x = [ basis.affine_mapping_1D( [-1, 1], [0, 1], xi ) for xi in x ]
#               self.assertAlmostEqual( first = basis.evalBernsteinBasisDeriv( degree = 2, basis_idx = 0, deriv = 0, domain = [0, 1], variate = x[0] ), second = 0.62200846792814620, delta = 1e-12 )
#               self.assertAlmostEqual( first = basis.evalBernsteinBasisDeriv( degree = 2, basis_idx = 0, deriv = 0, domain = [0, 1], variate = x[1] ), second = 0.04465819873852045, delta = 1e-12 )
#               self.assertAlmostEqual( first = basis.evalBernsteinBasisDeriv( degree = 2, basis_idx = 1, deriv = 0, domain = [0, 1], variate = x[0] ), second = 0.33333333333333333, delta = 1e-12 )
#               self.assertAlmostEqual( first = basis.evalBernsteinBasisDeriv( degree = 2, basis_idx = 1, deriv = 0, domain = [0, 1], variate = x[1] ), second = 0.33333333333333333, delta = 1e-12 )
#               self.assertAlmostEqual( first = basis.evalBernsteinBasisDeriv( degree = 2, basis_idx = 2, deriv = 0, domain = [0, 1], variate = x[0] ), second = 0.04465819873852045, delta = 1e-12 )
#               self.assertAlmostEqual( first = basis.evalBernsteinBasisDeriv( degree = 2, basis_idx = 2, deriv = 0, domain = [0, 1], variate = x[1] ), second = 0.62200846792814620, delta = 1e-12 )

#        def test_linear_1st_deriv_at_nodes( self ):
#               self.assertAlmostEqual( first = basis.evalBernsteinBasisDeriv( degree = 1, basis_idx = 0, deriv = 1, domain = [0, 1], variate = 0.0 ), second = -1.0, delta = 1e-12 )
#               self.assertAlmostEqual( first = basis.evalBernsteinBasisDeriv( degree = 1, basis_idx = 0, deriv = 1, domain = [0, 1], variate = 1.0 ), second = -1.0, delta = 1e-12 )
#               self.assertAlmostEqual( first = basis.evalBernsteinBasisDeriv( degree = 1, basis_idx = 1, deriv = 1, domain = [0, 1], variate = 0.0 ), second = +1.0, delta = 1e-12 )
#               self.assertAlmostEqual( first = basis.evalBernsteinBasisDeriv( degree = 1, basis_idx = 1, deriv = 1, domain = [0, 1], variate = 1.0 ), second = +1.0, delta = 1e-12 )

#        def test_linear_1st_deriv_at_gauss_pts( self ):
#               self.assertAlmostEqual( first = basis.evalBernsteinBasisDeriv( degree = 1, basis_idx = 0, deriv = 1, domain = [0, 1], variate = 0.5 ), second = -1.0, delta = 1e-12 )
#               self.assertAlmostEqual( first = basis.evalBernsteinBasisDeriv( degree = 1, basis_idx = 1, deriv = 1, domain = [0, 1], variate = 0.5 ), second = +1.0, delta = 1e-12 )

#        def test_linear_2nd_deriv_at_nodes( self ):
#               self.assertAlmostEqual( first = basis.evalBernsteinBasisDeriv( degree = 1, basis_idx = 0, deriv = 2, domain = [0, 1], variate = 0.0 ), second = 0, delta = 1e-12 )
#               self.assertAlmostEqual( first = basis.evalBernsteinBasisDeriv( degree = 1, basis_idx = 0, deriv = 2, domain = [0, 1], variate = 1.0 ), second = 0, delta = 1e-12 )
#               self.assertAlmostEqual( first = basis.evalBernsteinBasisDeriv( degree = 1, basis_idx = 1, deriv = 2, domain = [0, 1], variate = 0.0 ), second = 0, delta = 1e-12 )
#               self.assertAlmostEqual( first = basis.evalBernsteinBasisDeriv( degree = 1, basis_idx = 1, deriv = 2, domain = [0, 1], variate = 1.0 ), second = 0, delta = 1e-12 )

#        def test_linear_2nd_deriv_at_gauss_pts( self ):
#               self.assertAlmostEqual( first = basis.evalBernsteinBasisDeriv( degree = 1, basis_idx = 0, deriv = 2, domain = [0, 1], variate = 0.5 ), second = 0, delta = 1e-12 )
#               self.assertAlmostEqual( first = basis.evalBernsteinBasisDeriv( degree = 1, basis_idx = 1, deriv = 2, domain = [0, 1], variate = 0.5 ), second = 0, delta = 1e-12 )

#        def test_quadratic_1st_deriv_at_nodes( self ):
#               self.assertAlmostEqual( first = basis.evalBernsteinBasisDeriv( degree = 2, basis_idx = 0, deriv = 1, domain = [0, 1], variate = 0.0 ), second = -2.0, delta = 1e-12 )
#               self.assertAlmostEqual( first = basis.evalBernsteinBasisDeriv( degree = 2, basis_idx = 0, deriv = 1, domain = [0, 1], variate = 0.5 ), second = -1.0, delta = 1e-12 )
#               self.assertAlmostEqual( first = basis.evalBernsteinBasisDeriv( degree = 2, basis_idx = 0, deriv = 1, domain = [0, 1], variate = 1.0 ), second =  0.0, delta = 1e-12 )
#               self.assertAlmostEqual( first = basis.evalBernsteinBasisDeriv( degree = 2, basis_idx = 1, deriv = 1, domain = [0, 1], variate = 0.0 ), second = +2.0, delta = 1e-12 )
#               self.assertAlmostEqual( first = basis.evalBernsteinBasisDeriv( degree = 2, basis_idx = 1, deriv = 1, domain = [0, 1], variate = 0.5 ), second =  0.0, delta = 1e-12 )
#               self.assertAlmostEqual( first = basis.evalBernsteinBasisDeriv( degree = 2, basis_idx = 1, deriv = 1, domain = [0, 1], variate = 1.0 ), second = -2.0, delta = 1e-12 )
#               self.assertAlmostEqual( first = basis.evalBernsteinBasisDeriv( degree = 2, basis_idx = 2, deriv = 1, domain = [0, 1], variate = 0.0 ), second =  0.0, delta = 1e-12 )
#               self.assertAlmostEqual( first = basis.evalBernsteinBasisDeriv( degree = 2, basis_idx = 2, deriv = 1, domain = [0, 1], variate = 0.5 ), second =  1.0, delta = 1e-12 )
#               self.assertAlmostEqual( first = basis.evalBernsteinBasisDeriv( degree = 2, basis_idx = 2, deriv = 1, domain = [0, 1], variate = 1.0 ), second = +2.0, delta = 1e-12 )

#        def test_quadratic_1st_deriv_at_gauss_pts( self ):
#               x = [ -1.0 / math.sqrt(3.0) , 1.0 / math.sqrt(3.0) ]
#               x = [ basis.affine_mapping_1D( [-1, 1], [0, 1], xi ) for xi in x ]
#               self.assertAlmostEqual( first = basis.evalBernsteinBasisDeriv( degree = 2, basis_idx = 0, deriv = 1, domain = [0, 1], variate = x[0] ), second = -1.0 - 1/( math.sqrt(3) ), delta = 1e-12 )
#               self.assertAlmostEqual( first = basis.evalBernsteinBasisDeriv( degree = 2, basis_idx = 0, deriv = 1, domain = [0, 1], variate = x[1] ), second = -1.0 + 1/( math.sqrt(3) ), delta = 1e-12 )
#               self.assertAlmostEqual( first = basis.evalBernsteinBasisDeriv( degree = 2, basis_idx = 1, deriv = 1, domain = [0, 1], variate = x[0] ), second = +2.0 / math.sqrt(3), delta = 1e-12 )
#               self.assertAlmostEqual( first = basis.evalBernsteinBasisDeriv( degree = 2, basis_idx = 1, deriv = 1, domain = [0, 1], variate = x[1] ), second = -2.0 / math.sqrt(3), delta = 1e-12 )
#               self.assertAlmostEqual( first = basis.evalBernsteinBasisDeriv( degree = 2, basis_idx = 2, deriv = 1, domain = [0, 1], variate = x[0] ), second = +1.0 - 1/( math.sqrt(3) ), delta = 1e-12 )
#               self.assertAlmostEqual( first = basis.evalBernsteinBasisDeriv( degree = 2, basis_idx = 2, deriv = 1, domain = [0, 1], variate = x[1] ), second = +1.0 + 1/( math.sqrt(3) ), delta = 1e-12 )

#        def test_quadratic_2nd_deriv_at_nodes( self ):
#               self.assertAlmostEqual( first = basis.evalBernsteinBasisDeriv( degree = 2, basis_idx = 0, deriv = 1, domain = [0, 1], variate = 0.0 ), second = -2.0, delta = 1e-12 )
#               self.assertAlmostEqual( first = basis.evalBernsteinBasisDeriv( degree = 2, basis_idx = 0, deriv = 1, domain = [0, 1], variate = 0.5 ), second = -1.0, delta = 1e-12 )
#               self.assertAlmostEqual( first = basis.evalBernsteinBasisDeriv( degree = 2, basis_idx = 0, deriv = 1, domain = [0, 1], variate = 1.0 ), second =  0.0, delta = 1e-12 )
#               self.assertAlmostEqual( first = basis.evalBernsteinBasisDeriv( degree = 2, basis_idx = 1, deriv = 1, domain = [0, 1], variate = 0.0 ), second = +2.0, delta = 1e-12 )
#               self.assertAlmostEqual( first = basis.evalBernsteinBasisDeriv( degree = 2, basis_idx = 1, deriv = 1, domain = [0, 1], variate = 0.5 ), second =  0.0, delta = 1e-12 )
#               self.assertAlmostEqual( first = basis.evalBernsteinBasisDeriv( degree = 2, basis_idx = 1, deriv = 1, domain = [0, 1], variate = 1.0 ), second = -2.0, delta = 1e-12 )
#               self.assertAlmostEqual( first = basis.evalBernsteinBasisDeriv( degree = 2, basis_idx = 2, deriv = 1, domain = [0, 1], variate = 0.0 ), second =  0.0, delta = 1e-12 )
#               self.assertAlmostEqual( first = basis.evalBernsteinBasisDeriv( degree = 2, basis_idx = 2, deriv = 1, domain = [0, 1], variate = 0.5 ), second =  1.0, delta = 1e-12 )
#               self.assertAlmostEqual( first = basis.evalBernsteinBasisDeriv( degree = 2, basis_idx = 2, deriv = 1, domain = [0, 1], variate = 1.0 ), second = +2.0, delta = 1e-12 )

#        def test_quadratic_2nd_deriv_at_gauss_pts( self ):
#               x = [ -1.0 / math.sqrt(3.0) , 1.0 / math.sqrt(3.0) ]
#               x = [ basis.affine_mapping_1D( [-1, 1], [0, 1], xi ) for xi in x ]
#               self.assertAlmostEqual( first = basis.evalBernsteinBasisDeriv( degree = 2, basis_idx = 0, deriv = 2, domain = [0, 1], variate = x[0] ), second = +2.0, delta = 1e-12 )
#               self.assertAlmostEqual( first = basis.evalBernsteinBasisDeriv( degree = 2, basis_idx = 0, deriv = 2, domain = [0, 1], variate = x[1] ), second = +2.0, delta = 1e-12 )
#               self.assertAlmostEqual( first = basis.evalBernsteinBasisDeriv( degree = 2, basis_idx = 1, deriv = 2, domain = [0, 1], variate = x[0] ), second = -4.0, delta = 1e-12 )
#               self.assertAlmostEqual( first = basis.evalBernsteinBasisDeriv( degree = 2, basis_idx = 1, deriv = 2, domain = [0, 1], variate = x[1] ), second = -4.0, delta = 1e-12 )
#               self.assertAlmostEqual( first = basis.evalBernsteinBasisDeriv( degree = 2, basis_idx = 2, deriv = 2, domain = [0, 1], variate = x[0] ), second = +2.0, delta = 1e-12 )
#               self.assertAlmostEqual( first = basis.evalBernsteinBasisDeriv( degree = 2, basis_idx = 2, deriv = 2, domain = [0, 1], variate = x[1] ), second = +2.0, delta = 1e-12 )
              
# class Test_evalSplineBasisDeriv1D( unittest.TestCase ):
#         def test_C0_linear_0th_deriv_at_nodes( self ):
#               C = numpy.eye( 2 )
#               self.assertAlmostEqual( first = basis.evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 0, deriv = 0, domain = [ 0, 1 ], variate = 0.0 ), second = 1.0 )
#               self.assertAlmostEqual( first = basis.evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 1, deriv = 0, domain = [ 0, 1 ], variate = 0.0 ), second = 0.0 )
#               self.assertAlmostEqual( first = basis.evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 0, deriv = 0, domain = [ 0, 1 ], variate = 1.0 ), second = 0.0 )
#               self.assertAlmostEqual( first = basis.evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 1, deriv = 0, domain = [ 0, 1 ], variate = 1.0 ), second = 1.0 )

#         def test_C0_linear_1st_deriv_at_nodes( self ):
#               C = numpy.eye( 2 )
#               self.assertAlmostEqual( first = basis.evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 0, deriv = 1, domain = [ 0, 1 ], variate = 0.0 ), second = -1.0 )
#               self.assertAlmostEqual( first = basis.evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 1, deriv = 1, domain = [ 0, 1 ], variate = 0.0 ), second = +1.0 )
#               self.assertAlmostEqual( first = basis.evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 0, deriv = 1, domain = [ 0, 1 ], variate = 1.0 ), second = -1.0 )
#               self.assertAlmostEqual( first = basis.evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 1, deriv = 1, domain = [ 0, 1 ], variate = 1.0 ), second = +1.0 )

#         def test_C1_quadratic_0th_deriv_at_nodes( self ):
#               C = numpy.array( [ [ 1.0, 0.0, 0.0 ], [ 0.0, 1.0, 0.5 ], [ 0.0, 0.0, 0.5 ] ] )
#               self.assertAlmostEqual( first = basis.evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 0, deriv = 0, domain = [ 0, 1 ], variate = 0.0 ), second = 1.0 )
#               self.assertAlmostEqual( first = basis.evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 1, deriv = 0, domain = [ 0, 1 ], variate = 0.0 ), second = 0.0 )
#               self.assertAlmostEqual( first = basis.evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 2, deriv = 0, domain = [ 0, 1 ], variate = 0.0 ), second = 0.0 )
#               self.assertAlmostEqual( first = basis.evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 0, deriv = 0, domain = [ 0, 1 ], variate = 0.5 ), second = 0.25 )
#               self.assertAlmostEqual( first = basis.evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 1, deriv = 0, domain = [ 0, 1 ], variate = 0.5 ), second = 0.625 )
#               self.assertAlmostEqual( first = basis.evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 2, deriv = 0, domain = [ 0, 1 ], variate = 0.5 ), second = 0.125 )
#               self.assertAlmostEqual( first = basis.evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 0, deriv = 0, domain = [ 0, 1 ], variate = 1.0 ), second = 0.0 )
#               self.assertAlmostEqual( first = basis.evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 1, deriv = 0, domain = [ 0, 1 ], variate = 1.0 ), second = 0.5 )
#               self.assertAlmostEqual( first = basis.evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 2, deriv = 0, domain = [ 0, 1 ], variate = 1.0 ), second = 0.5 )

#         def test_C1_quadratic_1st_deriv_at_nodes( self ):
#               C = numpy.array( [ [ 1.0, 0.0, 0.0 ], [ 0.0, 1.0, 0.5 ], [ 0.0, 0.0, 0.5 ] ] )
#               self.assertAlmostEqual( first = basis.evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 0, deriv = 1, domain = [ 0, 1 ], variate = 0.0 ), second = -2.0 )
#               self.assertAlmostEqual( first = basis.evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 1, deriv = 1, domain = [ 0, 1 ], variate = 0.0 ), second = +2.0 )
#               self.assertAlmostEqual( first = basis.evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 2, deriv = 1, domain = [ 0, 1 ], variate = 0.0 ), second = +0.0 )
#               self.assertAlmostEqual( first = basis.evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 0, deriv = 1, domain = [ 0, 1 ], variate = 0.5 ), second = -1.0 )
#               self.assertAlmostEqual( first = basis.evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 1, deriv = 1, domain = [ 0, 1 ], variate = 0.5 ), second = +0.5 )
#               self.assertAlmostEqual( first = basis.evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 2, deriv = 1, domain = [ 0, 1 ], variate = 0.5 ), second = +0.5 )
#               self.assertAlmostEqual( first = basis.evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 0, deriv = 1, domain = [ 0, 1 ], variate = 1.0 ), second = +0.0 )
#               self.assertAlmostEqual( first = basis.evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 1, deriv = 1, domain = [ 0, 1 ], variate = 1.0 ), second = -1.0 )
#               self.assertAlmostEqual( first = basis.evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 2, deriv = 1, domain = [ 0, 1 ], variate = 1.0 ), second = +1.0 )

#         def test_C1_quadratic_2nd_deriv_at_nodes( self ):
#               C = numpy.array( [ [ 1.0, 0.0, 0.0 ], [ 0.0, 1.0, 0.5 ], [ 0.0, 0.0, 0.5 ] ] )
#               self.assertAlmostEqual( first = basis.evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 0, deriv = 2, domain = [ 0, 1 ], variate = 0.0 ), second = +2.0 )
#               self.assertAlmostEqual( first = basis.evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 1, deriv = 2, domain = [ 0, 1 ], variate = 0.0 ), second = -3.0 )
#               self.assertAlmostEqual( first = basis.evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 2, deriv = 2, domain = [ 0, 1 ], variate = 0.0 ), second = +1.0 )
#               self.assertAlmostEqual( first = basis.evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 0, deriv = 2, domain = [ 0, 1 ], variate = 0.5 ), second = +2.0 )
#               self.assertAlmostEqual( first = basis.evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 1, deriv = 2, domain = [ 0, 1 ], variate = 0.5 ), second = -3.0 )
#               self.assertAlmostEqual( first = basis.evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 2, deriv = 2, domain = [ 0, 1 ], variate = 0.5 ), second = +1.0 )
#               self.assertAlmostEqual( first = basis.evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 0, deriv = 2, domain = [ 0, 1 ], variate = 1.0 ), second = +2.0 )
#               self.assertAlmostEqual( first = basis.evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 1, deriv = 2, domain = [ 0, 1 ], variate = 1.0 ), second = -3.0 )
#               self.assertAlmostEqual( first = basis.evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 2, deriv = 2, domain = [ 0, 1 ], variate = 1.0 ), second = +1.0 )

#         def test_biunit_C0_linear_0th_deriv_at_nodes( self ):
#               C = numpy.eye( 2 )
#               self.assertAlmostEqual( first = basis.evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 0, deriv = 0, domain = [ -1, 1 ], variate = -1.0 ), second = 1.0 )
#               self.assertAlmostEqual( first = basis.evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 1, deriv = 0, domain = [ -1, 1 ], variate = -1.0 ), second = 0.0 )
#               self.assertAlmostEqual( first = basis.evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 0, deriv = 0, domain = [ -1, 1 ], variate = +1.0 ), second = 0.0 )
#               self.assertAlmostEqual( first = basis.evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 1, deriv = 0, domain = [ -1, 1 ], variate = +1.0 ), second = 1.0 )

#         def test_biunit_C0_linear_1st_deriv_at_nodes( self ):
#               C = numpy.eye( 2 )
#               self.assertAlmostEqual( first = basis.evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 0, deriv = 1, domain = [ -1, 1 ], variate = -1.0 ), second = -0.5 )
#               self.assertAlmostEqual( first = basis.evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 1, deriv = 1, domain = [ -1, 1 ], variate = -1.0 ), second = +0.5 )
#               self.assertAlmostEqual( first = basis.evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 0, deriv = 1, domain = [ -1, 1 ], variate = +1.0 ), second = -0.5 )
#               self.assertAlmostEqual( first = basis.evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 1, deriv = 1, domain = [ -1, 1 ], variate = +1.0 ), second = +0.5 )

#         def test_biunit_C1_quadratic_0th_deriv_at_nodes( self ):
#               C = numpy.array( [ [ 1.0, 0.0, 0.0 ], [ 0.0, 1.0, 0.5 ], [ 0.0, 0.0, 0.5 ] ] )
#               self.assertAlmostEqual( first = basis.evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 0, deriv = 0, domain = [ -1, 1 ], variate = -1.0 ), second = 1.0 )
#               self.assertAlmostEqual( first = basis.evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 1, deriv = 0, domain = [ -1, 1 ], variate = -1.0 ), second = 0.0 )
#               self.assertAlmostEqual( first = basis.evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 2, deriv = 0, domain = [ -1, 1 ], variate = -1.0 ), second = 0.0 )
#               self.assertAlmostEqual( first = basis.evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 0, deriv = 0, domain = [ -1, 1 ], variate = +0.0 ), second = 0.25 )
#               self.assertAlmostEqual( first = basis.evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 1, deriv = 0, domain = [ -1, 1 ], variate = +0.0 ), second = 0.625 )
#               self.assertAlmostEqual( first = basis.evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 2, deriv = 0, domain = [ -1, 1 ], variate = +0.0 ), second = 0.125 )
#               self.assertAlmostEqual( first = basis.evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 0, deriv = 0, domain = [ -1, 1 ], variate = +1.0 ), second = 0.0 )
#               self.assertAlmostEqual( first = basis.evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 1, deriv = 0, domain = [ -1, 1 ], variate = +1.0 ), second = 0.5 )
#               self.assertAlmostEqual( first = basis.evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 2, deriv = 0, domain = [ -1, 1 ], variate = +1.0 ), second = 0.5 )

#         def test_biunit_C1_quadratic_1st_deriv_at_nodes( self ):
#               C = numpy.array( [ [ 1.0, 0.0, 0.0 ], [ 0.0, 1.0, 0.5 ], [ 0.0, 0.0, 0.5 ] ] )
#               self.assertAlmostEqual( first = basis.evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 0, deriv = 1, domain = [ -1, 1 ], variate = -1.0 ), second = -1.0 )
#               self.assertAlmostEqual( first = basis.evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 1, deriv = 1, domain = [ -1, 1 ], variate = -1.0 ), second = +1.0 )
#               self.assertAlmostEqual( first = basis.evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 2, deriv = 1, domain = [ -1, 1 ], variate = -1.0 ), second = +0.0 )
#               self.assertAlmostEqual( first = basis.evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 0, deriv = 1, domain = [ -1, 1 ], variate = +0.0 ), second = -0.5 )
#               self.assertAlmostEqual( first = basis.evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 1, deriv = 1, domain = [ -1, 1 ], variate = +0.0 ), second = +0.25 )
#               self.assertAlmostEqual( first = basis.evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 2, deriv = 1, domain = [ -1, 1 ], variate = +0.0 ), second = +0.25 )
#               self.assertAlmostEqual( first = basis.evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 0, deriv = 1, domain = [ -1, 1 ], variate = +1.0 ), second = +0.0 )
#               self.assertAlmostEqual( first = basis.evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 1, deriv = 1, domain = [ -1, 1 ], variate = +1.0 ), second = -0.5 )
#               self.assertAlmostEqual( first = basis.evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 2, deriv = 1, domain = [ -1, 1 ], variate = +1.0 ), second = +0.5 )

#         def test_biunit_C1_quadratic_2nd_deriv_at_nodes( self ):
#               C = numpy.array( [ [ 1.0, 0.0, 0.0 ], [ 0.0, 1.0, 0.5 ], [ 0.0, 0.0, 0.5 ] ] )
#               self.assertAlmostEqual( first = basis.evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 0, deriv = 2, domain = [ -1, 1 ], variate = -1.0 ), second = +0.50 )
#               self.assertAlmostEqual( first = basis.evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 1, deriv = 2, domain = [ -1, 1 ], variate = -1.0 ), second = -0.75 )
#               self.assertAlmostEqual( first = basis.evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 2, deriv = 2, domain = [ -1, 1 ], variate = -1.0 ), second = +0.25 )
#               self.assertAlmostEqual( first = basis.evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 0, deriv = 2, domain = [ -1, 1 ], variate = +0.0 ), second = +0.50 )
#               self.assertAlmostEqual( first = basis.evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 1, deriv = 2, domain = [ -1, 1 ], variate = +0.0 ), second = -0.75 )
#               self.assertAlmostEqual( first = basis.evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 2, deriv = 2, domain = [ -1, 1 ], variate = +0.0 ), second = +0.25 )
#               self.assertAlmostEqual( first = basis.evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 0, deriv = 2, domain = [ -1, 1 ], variate = +1.0 ), second = +0.50 )
#               self.assertAlmostEqual( first = basis.evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 1, deriv = 2, domain = [ -1, 1 ], variate = +1.0 ), second = -0.75 )
#               self.assertAlmostEqual( first = basis.evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 2, deriv = 2, domain = [ -1, 1 ], variate = +1.0 ), second = +0.25 )      

class test_assembleStressMatrix( unittest.TestCase ):
        def test_one_linear_C0_element( self ):
              problem = { "elastic_modulus": 100,
                      "area": 0.01,
                      "length": 1.0,
                      "traction": { "value": 1e-3, "position": 1.0 },
                      "displacement": { "value": 0.0, "position": 0.0 },
                      "body_force": 0.0 }
              spline_space = { "domain": [0, problem[ "length" ]], "degree": [ 1 ], "continuity": [ -1, -1 ] }
              continuity = spline_space["continuity"]
              uspline.make_uspline_mesh( spline_space, "temp_uspline" )
              uspline_bext = bext.readBEXT( "temp_uspline.json" )
              test_stiffness_matrix = assembleStiffnessMatrix(continuity, problem = problem, uspline_bext = uspline_bext )
              gold_stiffness_matrix = numpy.array( [ [ 1.0, -1.0 ], [ -1.0, 1.0 ] ] )
              self.assertTrue( numpy.allclose( test_stiffness_matrix, gold_stiffness_matrix ) )

        def test_two_linear_C0_element( self ):
              problem = { "elastic_modulus": 100,
                      "area": 0.01,
                      "length": 1.0,
                      "traction": { "value": 1e-3, "position": 1.0 },
                      "displacement": { "value": 0.0, "position": 0.0 },
                      "body_force": 0.0 }
              spline_space = { "domain": [0, problem[ "length" ]], "degree": [ 1, 1 ], "continuity": [ -1, 0, -1 ] }
              continuity = spline_space["continuity"]
              uspline.make_uspline_mesh( spline_space, "temp_uspline" )
              uspline_bext = bext.readBEXT( "temp_uspline.json" )
              test_stiffness_matrix = assembleStiffnessMatrix(continuity, problem = problem, uspline_bext = uspline_bext )
              gold_stiffness_matrix = numpy.array( [ [ 2.0, -2.0, 0.0 ], [ -2.0, 4.0, -2.0 ], [ 0.0, -2.0, 2.0 ] ] )
              self.assertTrue( numpy.allclose( test_stiffness_matrix, gold_stiffness_matrix ) )

        def test_one_quadratic_C0_element( self ):
              problem = { "elastic_modulus": 100,
                      "area": 0.01,
                      "length": 1.0,
                      "traction": { "value": 1e-3, "position": 1.0 },
                      "displacement": { "value": 0.0, "position": 0.0 },
                      "body_force": 0.0 }
              spline_space = { "domain": [0, problem[ "length" ]], "degree": [ 2 ], "continuity": [ -1, -1 ] }
              continuity = spline_space["continuity"]
              uspline.make_uspline_mesh( spline_space, "temp_uspline" )
              uspline_bext = bext.readBEXT( "temp_uspline.json" )
              test_stiffness_matrix = assembleStiffnessMatrix( continuity, problem = problem, uspline_bext = uspline_bext)
              gold_stiffness_matrix = numpy.array( [ [  4.0 / 3.0, -2.0 / 3.0, -2.0 / 3.0 ],
                                                  [ -2.0 / 3.0,  4.0 / 3.0, -2.0 / 3.0 ],
                                                  [ -2.0 / 3.0, -2.0 / 3.0,  4.0 / 3.0 ] ] )
              self.assertTrue( numpy.allclose( test_stiffness_matrix, gold_stiffness_matrix ) )

        def test_two_quadratic_C0_element( self ):
              problem = { "elastic_modulus": 100,
                      "area": 0.01,
                      "length": 1.0,
                      "traction": { "value": 1e-3, "position": 1.0 },
                      "displacement": { "value": 0.0, "position": 0.0 },
                      "body_force": 0.0 }
              spline_space = { "domain": [0, problem[ "length" ]], "degree": [ 2, 2 ], "continuity": [ -1, 0, -1 ] }
              continuity = spline_space["continuity"]
              uspline.make_uspline_mesh( spline_space, "temp_uspline" )
              uspline_bext = bext.readBEXT( "temp_uspline.json" )
              test_stiffness_matrix = assembleStiffnessMatrix(continuity, problem = problem, uspline_bext = uspline_bext )
              gold_stiffness_matrix = numpy.array( [ [  8.0 / 3.0, -4.0 / 3.0, -4.0 / 3.0,  0.0,        0.0 ],
                                                  [ -4.0 / 3.0,  8.0 / 3.0, -4.0 / 3.0,  0.0,        0.0 ],
                                                  [ -4.0 / 3.0, -4.0 / 3.0, 16.0 / 3.0, -4.0 / 3.0, -4.0 / 3.0 ],
                                                  [  0.0,        0.0,       -4.0 / 3.0,  8.0 / 3.0, -4.0 / 3.0 ],
                                                  [  0.0,        0.0,       -4.0 / 3.0, -4.0 / 3.0,  8.0 / 3.0 ] ] )
              self.assertTrue( numpy.allclose( test_stiffness_matrix, gold_stiffness_matrix ) )

        def test_two_quadratic_C1_element( self ):
              problem = { "elastic_modulus": 100,
                      "area": 0.01,
                      "length": 1.0,
                      "traction": { "value": 1e-3, "position": 1.0 },
                      "displacement": { "value": 0.0, "position": 0.0 },
                      "body_force": 0.0 }
              spline_space = { "domain": [0, problem[ "length" ]], "degree": [ 2, 2 ], "continuity": [ -1, 1, -1 ] }
              continuity = spline_space["continuity"]
              uspline.make_uspline_mesh( spline_space, "temp_uspline" )
              uspline_bext = bext.readBEXT( "temp_uspline.json" )
              test_stiffness_matrix = assembleStiffnessMatrix(continuity, problem = problem, uspline_bext = uspline_bext )
              gold_stiffness_matrix = numpy.array( [ [  8.0 / 3.0, -2.0,       -2.0/ 3.0,   0.0 ],
                                                  [ -2.0,        8.0 / 3.0,  0.0,       -2.0 / 3.0 ],
                                                  [ -2.0 / 3.0,  0.0,        8.0 / 3.0, -2.0 ],
                                                  [  0.0,       -2.0 / 3.0, -2.0,        8.0 / 3.0 ] ] )
              self.assertTrue( numpy.allclose( test_stiffness_matrix, gold_stiffness_matrix ) )

class test_ComputeSolution( unittest.TestCase ):
    def test_simple( self ):
           problem = { "elastic_modulus": 100,
                       "area": 0.01,
                       "length": 1.0,
                       "traction": { "value": 1e-3, "position": 1.0 },
                       "displacement": { "value": 0.0, "position": 0.0 },
                       "body_force": 1e-3 }
           spline_space = { "domain": [0, problem[ "length" ]], "degree": [ 1, 1, 1 ], "continuity": [ -1, 0, 0, -1 ] }
           continuity = spline_space["continuity"]
           uspline.make_uspline_mesh( spline_space, "temp_uspline" )
           uspline_bext = bext.readBEXT( "temp_uspline.json" )
           test_sol_coeff = computeSolution(continuity, problem = problem, uspline_bext = uspline_bext )
           gold_sol_coeff = numpy.array( [ 0.0, 11.0 / 18000.0, 1.0 / 900.0, 3.0 / 2000.0 ] )
           self.assertTrue( numpy.allclose( test_sol_coeff, gold_sol_coeff ) )
           plotSolution( test_sol_coeff, uspline_bext )
           plotCompareFunToExactSolution( problem, test_sol_coeff, uspline_bext )

    def test_textbook_problem( self ):
           problem = { "elastic_modulus": 200e9,
                       "area": 1.0,
                       "length": 5.0,
                       "traction": { "value": 9810.0, "position": 5.0 },
                       "displacement": { "value": 0.0, "position": 0.0 },
                       "body_force": 784800.0 }
           spline_space = { "domain": [0, problem[ "length" ]], "degree": [ 2, 2 ], "continuity": [ -1, 1, -1 ] }
           continuity = spline_space["continuity"]
           uspline.make_uspline_mesh( spline_space, "temp_uspline" )
           uspline_bext = bext.readBEXT( "temp_uspline.json" )
           test_sol_coeff = computeSolution(continuity ,problem = problem, uspline_bext = uspline_bext )
           gold_sol_coeff = numpy.array( [0.0, 2.45863125e-05, 4.92339375e-05, 4.92952500e-05] )
           self.assertTrue( numpy.allclose( test_sol_coeff, gold_sol_coeff ) )
           plotSolution( test_sol_coeff, uspline_bext )
           plotCompareFunToExactSolution( problem, test_sol_coeff, uspline_bext )

unittest.main()