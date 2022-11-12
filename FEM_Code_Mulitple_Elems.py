# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 17:44:42 2022

@author: rphar
"""
import numpy
import scipy
from matplotlib import pyplot as plt
import unittest

import Basis_Func as basis
import Mesh_Generation as mesh
import Quadrature as quad

def Change_of_Domain(original_domain,output_domain,old_x):
    a = original_domain[0]
    b = original_domain[1]
    c = output_domain[0]
    d = output_domain[1]
    
    x = ((d-c)/(b-a))*old_x + ((b*c-a*d)/(b-a))
    
    return x

def Jacobian(original_domain,output_domain):
    a = original_domain[0]
    b = original_domain[1]
    c = output_domain[0]
    d = output_domain[1]
    
    jacobian = ((d-c)/(b-a))
        
    return jacobian

def Local_to_Global_Matrix(M_list,degree):
    dim = sum(degree) + 1
    M = numpy.zeros((dim,dim))
    
    for i in range(0,len(M_list)):
        for a in range(0,degree[i]+1):
            A = i * (degree[i]) + a 
            for b in range(0,degree[i]+1):
                B = i * (degree[i]) + b
                M[A,B] += M_list[i][a,b]
    
    return M

def Local_to_Global_F_Matrix(F_list,degree):
    dim = sum(degree) + 1
    F = numpy.zeros((dim))
    
    for i in range(0,len(F_list)):
        for a in range(0,degree[i]+1):
            A = i * (degree[i]) + a
            F[A] += F_list[i][a]
    
    return F

def assembleGramMatrix(node_coords, ien_array, solution_basis):
    num_elems = len(ien_array)
    degree_list = []
    M_list = []
    for i in range(0,num_elems):
        degree = int(len(ien_array[i])-1)
        degree_list.append(degree)
        n = int(numpy.ceil((2*degree+1)/2))
        xi_qp, w_qp = quad.computeGaussLegendreQuadrature(n)
        domain = [node_coords[i][0],node_coords[i][-1]]
        
        M_term = numpy.zeros((degree+1,degree+1))
        for A in range(0,degree+1):
            for B in range(0,degree+1):
                for q in range(0,len(xi_qp)):
                    
                    if solution_basis == basis.evalBernsteinBasis1D:
                        N_A = basis.evalBernsteinBasis1D(xi_qp[q], degree, A)
                        N_B = basis.evalBernsteinBasis1D(xi_qp[q], degree, B)
                        original_domain = [-1,1]
                        jacob = Jacobian(original_domain, output_domain = domain)
                    elif solution_basis == basis.evalLegendreBasis1D:
                        N_A = basis.evalLegendreBasis1D(A, xi_qp[q])
                        N_B = basis.evalLegendreBasis1D(B, xi_qp[q])
                        original_domain = [-1,1]
                        jacob = Jacobian(original_domain, output_domain = domain)
                    elif solution_basis == basis.evaluateLagrangeBasis1D:
                        N_A = basis.evaluateLagrangeBasis1D(xi_qp[q], degree, A)
                        N_B = basis.evaluateLagrangeBasis1D(xi_qp[q], degree, B)
                        original_domain = [-1,1]
                        jacob = Jacobian(original_domain, output_domain = domain)
                    M_term[A,B] += N_A * N_B * w_qp[q] * jacob
            
        M_list.append(M_term)

    M = Local_to_Global_Matrix(M_list, degree_list)
    
    return M

def assembleForceVector(target_fun, node_coords, ien_array, solution_basis):
    num_elems = len(ien_array)
    degree_list = []
    F_list = []
    
    for i in range(0,num_elems):
        degree = int(len(ien_array[i])-1)
        degree_list.append(degree)
        n = int(numpy.ceil((2*degree+1)/2))
        xi_pq, w_pq = quad.computeGaussLegendreQuadrature(n)
        domain = [node_coords[i][0],node_coords[i][-1]]
        
        F_term = numpy.zeros((degree+1))
        for A in range(0,degree+1):
            for q in range(0,len(xi_pq)):
                if solution_basis == basis.evalBernsteinBasis1D:
                    N_A = basis.evalBernsteinBasis1D(xi_pq[q], degree, A)
                    jacob = Jacobian([-1,1], domain)
                elif solution_basis == basis.evalLegendreBasis1D:
                    N_A = basis.evalLegendreBasis1D(A, xi_pq[q])
                    jacob = Jacobian([-1,1], domain)
                elif solution_basis == basis.evaluateLagrangeBasis1D:
                    N_A = basis.evaluateLagrangeBasis1D(xi_pq[q], degree, A)
                    jacob = Jacobian([-1,1], domain)
                
                F_term[A] += N_A * target_fun(Change_of_Domain([-1,1],domain,xi_pq[q])) * w_pq[q] * jacob 
        F_list.append(F_term)
    F = Local_to_Global_F_Matrix(F_list,degree_list)
    return F

def computeSolution(target_fun, domain, degree, solution_basis):
    xmin = domain[0]
    xmax = domain[1]
    node_coords,ien_array = mesh.generateMesh(xmin, xmax, degree)
    M = assembleGramMatrix(node_coords, ien_array, solution_basis)
    F = assembleForceVector(target_fun, node_coords, ien_array, solution_basis)
    F = F.transpose()
    d = numpy.linalg.solve(M, F)
    return d, node_coords, ien_array

def evaluateSolutionAt( x, d_matrix, node_coords, ien_array, eval_basis ):
    y = 0.0
    domain = [node_coords[0][0],node_coords[-1][-1]]
    num_elems = len(ien_array)
    
    #identifies element of x point
    elem_bound = numpy.linspace(domain[0],domain[1],num_elems+1)
    for i in range(0,num_elems):
        if x >= elem_bound[i] and x <= elem_bound[i+1]:
            elem_idx = i
        else:
            continue
    
    degree = len(ien_array[elem_idx])-1
    elem_domain = [node_coords[elem_idx][0],node_coords[elem_idx][-1]]
    param_x = Change_of_Domain(elem_domain,[-1,1],x)
    for i in range(0,len(ien_array[elem_idx])):
        d_idx = ien_array[elem_idx][i]
        if eval_basis == basis.evalLegendreBasis1D:
            func = eval_basis(i,param_x)
        else:
            func = eval_basis(param_x,degree,i)
        y += func * d_matrix[d_idx]
    
    return y

def computeFitError( target_fun, d_matrix, node_coords, ien_array, eval_basis ):
    num_elems = len(ien_array)
    xmin = node_coords[0][0]
    xmax = node_coords[-1][-1]
    domain = [xmin, xmax ]
    abs_err_fun = lambda x : abs( target_fun( x ) - evaluateSolutionAt( x, d_matrix, node_coords, ien_array, eval_basis ) )
    fit_error, residual = scipy.integrate.quad( abs_err_fun, domain[0], domain[1], epsrel = 1e-12, limit = num_elems * 100 )
    return fit_error, residual

def plotCompareGoldTestSolution( gold_sol_coeff, test_sol_coeff, node_coords, ien_array, solution_basis ):
    xmin = node_coords[0][0]
    xmax = node_coords[-1][-1]
    domain = [xmin, xmax ]
    x = numpy.linspace( domain[0], domain[1], 1000 )
    yg = numpy.zeros( 1000 )
    yt = numpy.zeros( 1000 )
    for i in range(0, len(x) ):
        yg[i] = evaluateSolutionAt( x[i], test_sol_coeff, node_coords, ien_array, solution_basis )
        yt[i] = evaluateSolutionAt( x[i], gold_sol_coeff, node_coords, ien_array, solution_basis )
    plt.plot( x, yg )
    plt.plot( x, yt )
    plt.show()

def plotCompareFunToTestSolution(  target_fun, test_sol_coeff, node_coords, ien_array, solution_basis ):
    domain = [node_coords[0][0],node_coords[-1][-1]]
    x = numpy.linspace( domain[0], domain[1], 1000 )
    y = numpy.zeros( 1000 )
    yt = numpy.zeros( 1000 )
    for i in range(0, len(x) ):
        y[i] = target_fun( x[i] )
        yt[i] = evaluateSolutionAt(x[i], test_sol_coeff, node_coords, ien_array, solution_basis )
    plt.plot( x, y )
    plt.plot( x, yt )
    plt.show()

class Test_computeSolution( unittest.TestCase ):
    def test_cubic_polynomial_target( self ):
        # print( "POLY TEST" )
        target_fun = lambda x: x**3 - (8/5)*x**2 + (3/5)*x
        domain = [ 0, 1 ]
        degree = [2]*2
        solution_basis = basis.evalBernsteinBasis1D
        test_sol_coeff, node_coords, ien_array = computeSolution( target_fun = target_fun, domain = domain, degree = degree, solution_basis =solution_basis )
        gold_sol_coeff = numpy.array( [ 1.0 / 120.0, 9.0 / 80.0, 1.0 / 40.0, -1.0 / 16.0, -1.0 / 120.0 ] )
        abs_err, rel_err = computeFitError( target_fun, test_sol_coeff, node_coords, ien_array, solution_basis )
        # plotCompareGoldTestSolution( gold_sol_coeff, test_sol_coeff, node_coords, ien_array, solution_basis )
        plotCompareFunToTestSolution( target_fun, test_sol_coeff, node_coords, ien_array, solution_basis )
        self.assertTrue( numpy.allclose( gold_sol_coeff, test_sol_coeff ) )
        self.assertAlmostEqual( first = rel_err, second = 0, delta = 1e-1 )

    def test_sin_target( self ):
        # print( "SIN TEST" )
        target_fun = lambda x: numpy.sin( numpy.pi * x )
        domain = [ 0, 1 ]
        degree = [2]*2
        solution_basis = basis.evalBernsteinBasis1D
        test_sol_coeff, node_coords, ien_array = computeSolution( target_fun = target_fun, domain = domain, degree = degree, solution_basis = solution_basis )
        gold_sol_coeff = numpy.array( [ -0.02607008, 0.9185523, 1.01739261, 0.9185523, -0.02607008 ] )
        abs_err, rel_err = computeFitError( target_fun, test_sol_coeff, node_coords, ien_array, solution_basis )
        # plotCompareGoldTestSolution( gold_sol_coeff, test_sol_coeff, node_coords, ien_array, solution_basis )
        plotCompareFunToTestSolution( target_fun, test_sol_coeff, node_coords, ien_array, solution_basis )
        self.assertAlmostEqual( first = rel_err, second = 0, delta = 1e-1 )
        
    def test_erfc_target( self ):
        # print( "ERFC TEST" )
        target_fun = lambda x: numpy.real( scipy.special.erfc( x ) )
        domain = [ -2, 2 ]
        degree = [3]*2
        solution_basis = basis.evalBernsteinBasis1D
        test_sol_coeff, node_coords, ien_array = computeSolution( target_fun = target_fun, domain = domain, degree = degree, solution_basis = solution_basis )
        gold_sol_coeff = numpy.array( [ 1.98344387, 2.0330054, 1.86372084, 1., 0.13627916, -0.0330054, 0.01655613 ] )
        abs_err, rel_err = computeFitError( target_fun, test_sol_coeff, node_coords, ien_array, solution_basis )
        # plotCompareGoldTestSolution( gold_sol_coeff, test_sol_coeff, node_coords, ien_array, solution_basis )
        plotCompareFunToTestSolution( target_fun, test_sol_coeff, node_coords, ien_array, solution_basis )
        self.assertAlmostEqual( first = rel_err, second = 0, delta = 1e-2 )
    
    def test_exptx_target( self ):
        # print( "EXPT TEST" )
        target_fun = lambda x: float( numpy.real( float( x )**float( x ) ) )
        domain = [ -1, 1 ]
        degree = [5]*2
        solution_basis = basis.evalBernsteinBasis1D
        test_sol_coeff, node_coords, ien_array = computeSolution( target_fun = target_fun, domain = domain, degree = degree, solution_basis = solution_basis )
        gold_sol_coeff = ( [ -1.00022471, -1.19005562, -0.9792369, 0.70884334, 1.73001439, 0.99212064, 0.44183573, 0.87014465, 0.5572111, 0.85241908, 0.99175228 ] )
        abs_err, rel_err = computeFitError( target_fun, test_sol_coeff, node_coords, ien_array, solution_basis )
        # plotCompareGoldTestSolution( gold_sol_coeff, test_sol_coeff, node_coords, ien_array, solution_basis )
        plotCompareFunToTestSolution( target_fun, test_sol_coeff, node_coords, ien_array, solution_basis )
        self.assertAlmostEqual( first = rel_err, second = 0, delta = 1e-2 )
        
class Test_assembleGramMatrix( unittest.TestCase ):
    def test_linear_lagrange( self ):
        domain = [ 0, 1 ]
        degree = [ 1, 1 ]
        node_coords, ien_array = mesh.generateMesh( domain[0], domain[1], degree )
        test_gram_matrix = assembleGramMatrix( node_coords = node_coords, ien_array = ien_array, solution_basis = basis.evaluateLagrangeBasis1D )
        gold_gram_matrix = numpy.array( [ [1/6, 1/12, 0], [1/12, 1/3, 1/12], [0, 1/12, 1/6] ] )
        self.assertTrue( numpy.allclose( test_gram_matrix, gold_gram_matrix ) )

    def test_quadratic_lagrange( self ):
        domain = [ 0, 1 ]
        degree = [ 2, 2 ]
        node_coords, ien_array = mesh.generateMesh( domain[0], domain[1], degree )
        test_gram_matrix = assembleGramMatrix( node_coords = node_coords, ien_array = ien_array, solution_basis = basis.evaluateLagrangeBasis1D )
        gold_gram_matrix = numpy.array( [ [1/15, 1/30, -1/60, 0, 0 ], [1/30, 4/15, 1/30, 0, 0], [-1/60, 1/30, 2/15, 1/30, -1/60], [ 0, 0, 1/30, 4/15, 1/30], [0, 0, -1/60, 1/30, 1/15] ] )
        self.assertTrue( numpy.allclose( test_gram_matrix, gold_gram_matrix ) )
    
    def test_cubic_lagrange( self ):
        domain = [ 0, 1 ]
        degree = [ 3, 3 ]
        node_coords, ien_array = mesh.generateMesh( domain[0], domain[1], degree )
        test_gram_matrix = assembleGramMatrix( node_coords = node_coords, ien_array = ien_array, solution_basis = basis.evaluateLagrangeBasis1D )
        gold_gram_matrix = numpy.array( [ [ 0.03809524,  0.02946429, -0.01071429,  0.00565476,  0.00000000,  0.00000000,  0.00000000 ], 
                                          [ 0.02946429,  0.19285714, -0.02410714, -0.01071429,  0.00000000,  0.00000000,  0.00000000 ], 
                                          [-0.01071429, -0.02410714,  0.19285714,  0.02946429,  0.00000000,  0.00000000,  0.00000000 ], 
                                          [ 0.00565476, -0.01071429,  0.02946429,  0.07619048,  0.02946429, -0.01071429,  0.00565476 ], 
                                          [ 0.00000000,  0.00000000,  0.00000000,  0.02946429,  0.19285714, -0.02410714, -0.01071429 ], 
                                          [ 0.00000000,  0.00000000,  0.00000000, -0.01071429, -0.02410714,  0.19285714,  0.02946429 ], 
                                          [ 0.00000000,  0.00000000,  0.00000000,  0.00565476, -0.01071429,  0.02946429,  0.03809524 ] ] )
        self.assertTrue( numpy.allclose( test_gram_matrix, gold_gram_matrix ) )
    
    def test_linear_bernstein( self ):
        domain = [ 0, 1 ]
        degree = [ 1, 1 ]
        node_coords, ien_array = mesh.generateMesh( domain[0], domain[1], degree )
        test_gram_matrix = assembleGramMatrix( node_coords = node_coords, ien_array = ien_array, solution_basis = basis.evalBernsteinBasis1D )
        gold_gram_matrix = numpy.array( [ [1/6, 1/12, 0], [1/12, 1/3, 1/12], [0, 1/12, 1/6] ] )
        self.assertTrue( numpy.allclose( test_gram_matrix, gold_gram_matrix ) )

    def test_quadratic_bernstein( self ):
        domain = [ 0, 1 ]
        degree = [ 2, 2 ]
        node_coords, ien_array = mesh.generateMesh( domain[0], domain[1], degree )
        test_gram_matrix = assembleGramMatrix( node_coords = node_coords, ien_array = ien_array, solution_basis = basis.evalBernsteinBasis1D )
        gold_gram_matrix = numpy.array( [ [1/10, 1/20, 1/60, 0, 0 ], [1/20, 1/15, 1/20, 0, 0 ], [1/60, 1/20, 1/5, 1/20, 1/60], [0, 0, 1/20, 1/15, 1/20], [0, 0, 1/60, 1/20, 1/10] ] )
        self.assertTrue( numpy.allclose( test_gram_matrix, gold_gram_matrix ) )
    
    def test_cubic_bernstein( self ):
        domain = [ 0, 1 ]
        degree = [ 3, 3 ]
        node_coords, ien_array = mesh.generateMesh( domain[0], domain[1], degree )
        test_gram_matrix = assembleGramMatrix( node_coords = node_coords, ien_array = ien_array, solution_basis = basis.evalBernsteinBasis1D )
        gold_gram_matrix = numpy.array( [ [1/14, 1/28, 1/70, 1/280, 0, 0, 0 ], [1/28, 3/70, 9/280, 1/70, 0, 0, 0 ], [1/70, 9/280, 3/70, 1/28, 0, 0, 0 ], [1/280, 1/70, 1/28, 1/7, 1/28, 1/70, 1/280], [0, 0, 0, 1/28, 3/70, 9/280, 1/70], [0, 0, 0, 1/70, 9/280, 3/70, 1/28], [0, 0, 0, 1/280, 1/70, 1/28, 1/14 ] ] )
        self.assertTrue( numpy.allclose( test_gram_matrix, gold_gram_matrix ) )
        
class Test_assembleForceVector( unittest.TestCase ):
    def test_lagrange_const_force_fun( self ):
        domain = [ 0, 1 ]
        degree = [ 3, 3 ]
        target_fun = lambda x: numpy.pi
        node_coords, ien_array = mesh.generateMesh( domain[0], domain[1], degree )
        test_force_vector = assembleForceVector( target_fun = target_fun, node_coords = node_coords, ien_array = ien_array, solution_basis = basis.evaluateLagrangeBasis1D )
        gold_force_vector = numpy.array( [ numpy.pi / 16.0, 3.0 * numpy.pi / 16.0, 3.0 * numpy.pi / 16.0, numpy.pi / 8.0, 3.0 * numpy.pi / 16.0, 3.0 * numpy.pi / 16.0, numpy.pi / 16.0 ] )
        self.assertTrue( numpy.allclose( test_force_vector, gold_force_vector ) )
    
    def test_lagrange_linear_force_fun( self ):
        domain = [ 0, 1 ]
        degree = [ 3, 3 ]
        target_fun = lambda x: 2*x + numpy.pi
        node_coords, ien_array = mesh.generateMesh( domain[0], domain[1], degree )
        test_force_vector = assembleForceVector( target_fun = target_fun, node_coords = node_coords, ien_array = ien_array, solution_basis = basis.evaluateLagrangeBasis1D )
        gold_force_vector = numpy.array( [ 0.20468287, 0.62654862, 0.73904862, 0.51769908, 0.81404862, 0.92654862, 0.31301621 ] )
        self.assertTrue( numpy.allclose( test_force_vector, gold_force_vector ) )
    
    def test_lagrange_quadratic_force_fun( self ):
        domain = [ 0, 1 ]
        degree = [ 3, 3 ]
        target_fun = lambda x: x**2.0
        node_coords, ien_array = mesh.generateMesh( domain[0], domain[1], degree )
        test_force_vector = assembleForceVector( target_fun = target_fun, node_coords = node_coords, ien_array = ien_array, solution_basis = basis.evaluateLagrangeBasis1D )
        gold_force_vector = numpy.array( [ 1.04166667e-03, 0, 2.81250000e-02, 3.33333333e-02, 6.56250000e-02, 1.50000000e-01, 5.52083333e-02 ] )
        self.assertTrue( numpy.allclose( test_force_vector, gold_force_vector ) )

    def test_lagrange_const_force_fun( self ):
        domain = [ 0, 1 ]
        degree = [ 3, 3 ]
        target_fun = lambda x: numpy.pi
        node_coords, ien_array = mesh.generateMesh( domain[0], domain[1], degree )
        test_force_vector = assembleForceVector( target_fun = target_fun, node_coords = node_coords, ien_array = ien_array, solution_basis = basis.evalBernsteinBasis1D )
        gold_force_vector = numpy.array( [ numpy.pi / 8.0, numpy.pi / 8.0, numpy.pi / 8.0, numpy.pi / 4.0, numpy.pi / 8.0, numpy.pi / 8.0, numpy.pi / 8.0 ] )
        self.assertTrue( numpy.allclose( test_force_vector, gold_force_vector ) )
    
    def test_bernstein_linear_force_fun( self ):
        domain = [ 0, 1 ]
        degree = [ 3, 3 ]
        target_fun = lambda x: 2*x + numpy.pi
        node_coords, ien_array = mesh.generateMesh( domain[0], domain[1], degree )
        test_force_vector = assembleForceVector( target_fun = target_fun, node_coords = node_coords, ien_array = ien_array, solution_basis = basis.evalBernsteinBasis1D )
        gold_force_vector = numpy.array( [ 0.41769908, 0.44269908, 0.46769908, 1.03539816, 0.56769908, 0.59269908, 0.61769908 ] )
        self.assertTrue( numpy.allclose( test_force_vector, gold_force_vector ) )
    
    def test_bernstein_quadratic_force_fun( self ):
        domain = [ 0, 1 ]
        degree = [ 3, 3 ]
        target_fun = lambda x: x**2.0
        node_coords, ien_array = mesh.generateMesh( domain[0], domain[1], degree )
        test_force_vector = assembleForceVector( target_fun = target_fun, node_coords = node_coords, ien_array = ien_array, solution_basis = basis.evalBernsteinBasis1D )
        gold_force_vector = numpy.array( [ 1/480, 1/160, 1/80, 1/15, 1/16, 13/160, 49/480 ] )
        self.assertTrue( numpy.allclose( test_force_vector, gold_force_vector ) )

unittest.main()