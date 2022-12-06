# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 17:22:29 2022

@author: rphar
"""
import numpy
import sympy
import scipy
import argparse
import unittest

import Basis_Func as basis
import Quadrature as quadrature

import uspline
import bext

def Jacobian(original_domain,output_domain):
    a = original_domain[0]
    b = original_domain[1]
    c = output_domain[0]
    d = output_domain[1]
    jacobian = ((d-c)/(b-a))        
    return jacobian

def Change_of_Domain(original_domain,output_domain,old_x):
    a = original_domain[0]
    b = original_domain[1]
    c = output_domain[0]
    d = output_domain[1]
    x = ((d-c)/(b-a))*old_x + ((b*c-a*d)/(b-a))    
    return x

def Local_to_Global_Matrix(M_list,degree,continuity):
    del continuity[0]
    del continuity[-1]
    dim = sum(degree) - sum(continuity) + 1
    M = numpy.zeros((dim,dim))
    for i in range(0,len(M_list)):
        for a in range(0,degree[i]+1):
            if i == 0:
                A = i * (degree[i]) + a
            else:
                A = i * (degree[i]) + a - continuity[i-1] 
            for b in range(0,degree[i]+1):
                if i == 0:
                    B = i * (degree[i]) + b
                else:
                    B = i * (degree[i]) + b - continuity[i-1]
                M[A,B] += M_list[i][a,b]
    return M

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

def assembleGramMatrix(continuity,uspline_bext):
    num_elems = bext.getNumElems(uspline_bext)
    degree_list = []
    M_list = []
    for i in range(0,num_elems):
        degree = bext.getElementDegree(uspline_bext, i)
        degree_list.append(degree)
        n = int(numpy.ceil((2*degree+1)/2))
        xi_qp, w_qp = quadrature.computeGaussLegendreQuadrature(n)
        domain = bext.getElementDomain(uspline_bext, i)
        extraction_operator = bext.getElementExtractionOperator(uspline_bext, i)
        
        M_term = numpy.zeros((degree+1,degree+1))
        jacob = Jacobian(original_domain = [-1,1], output_domain = domain)
        for A in range(0,degree+1):
            for B in range(0,degree+1):
                for q in range(0,len(xi_qp)):
                    N_A = basis.evalSplineBasis1D(extraction_operator, A, [-1,1], xi_qp[q])
                    N_B = basis.evalSplineBasis1D(extraction_operator, B, [-1,1], xi_qp[q])
                    M_term[A,B] += N_A * N_B * w_qp[q] * jacob
            
        M_list.append(M_term)

    M = Local_to_Global_Matrix(M_list, degree_list,continuity)
    return M

def assembleForceVector(continuity,target_fun, uspline_bext):
    num_elems =  bext.getNumElems(uspline_bext)
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
        jacob = Jacobian(original_domain = [-1,1], output_domain = domain)
        for A in range(0,degree+1):
            for q in range(0,len(xi_qp)):
                N_A = basis.evalSplineBasis1D(extraction_operator, A, domain, xi_qp[q])
                
                F_term[A] += N_A * target_fun(Change_of_Domain([-1,1],domain,xi_qp[q])) * w_qp[q] * jacob 
        F_list.append(F_term)
    F = Local_to_Global_F_Matrix(F_list,degree_list,continuity)
    return F

def computeSolution(target_fun, uspline_bext):
    M = assembleGramMatrix(uspline_bext)
    F = assembleForceVector(target_fun, uspline_bext)
    F = F.transpose()
    d = numpy.linalg.solve(M, F)
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

def computeElementFitError( target_fun, coeff, uspline_bext, elem_id ):
    domain = bext.getDomain( uspline_bext )
    elem_domain = bext.getElementDomain( uspline_bext, elem_id )
    elem_degree = bext.getElementDegree( uspline_bext, elem_id )
    num_qp = int( numpy.ceil( ( 2*elem_degree + 1 ) / 2.0 ) + 1 )
    abs_err_fun = lambda x : abs( target_fun( basis.affine_mapping_1D( [-1, 1], elem_domain, x ) ) - evaluateSolutionAt( basis.affine_mapping_1D( [-1, 1], elem_domain, x ), coeff, uspline_bext ) )
    abs_error = quadrature.quad( abs_err_fun, elem_domain, num_qp )
    return abs_error

def computeFitError( target_fun, coeff, uspline_bext ):
    num_elems = bext.getNumElems( uspline_bext )
    abs_error = 0.0
    for elem_idx in range( 0, num_elems ):
        elem_id = bext.elemIdFromElemIdx( uspline_bext, elem_idx )
        abs_error += computeElementFitError( target_fun, coeff, uspline_bext, elem_id )
    domain = bext.getDomain( uspline_bext )
    target_fun_norm, _ = scipy.integrate.quad( lambda x: abs( target_fun(x) ), domain[0], domain[1], epsrel = 1e-12, limit = num_elems * 100 )
    rel_error = abs_error / target_fun_norm
    return abs_error, rel_error

## CLI ARGUMENT PARSING
def prepareCommandInputs( target_fun_str, domain, degree, continuity ):
    spline_space = { "domain": domain, "degree": degree, "continuity": continuity }
    target_fun = sympy.parsing.sympy_parser.parse_expr( target_fun_str )
    target_fun = sympy.lambdify( sympy.symbols( "x", real = True ), target_fun )
    return target_fun, spline_space

def parseCommandLineArguments( ):
    parser = argparse.ArgumentParser()
    parser.add_argument( "–function", "-f",   nargs = 1,   type = str,   required = True )
    parser.add_argument( "–domain", "-d",     nargs = 2,   type = float, required = True )
    parser.add_argument( "–degree", "-p",     nargs = '+', type = int,   required = True )
    parser.add_argument( "–continuity", "-c", nargs = '+', type = int,   required = True )
    args = parser.parse_args( )
    return args.function[0], args.domain, args.degree, args.continuity

## TEST CALLING FROM PYTHON
# class Test_python( unittest.TestCase ):
#     def test_run( self ):
#         target_fun_str = "sin(pi*x)"
#         domain = [ 0, 1 ]
#         degree = [ 2, 2 ]
#         continuity = [ -1, 1, -1 ]
#         target_fun, spline_space = prepareCommandInputs( target_fun_str, domain, degree, continuity )
#         sol = main( target_fun, spline_space )

# # EXAMPLE USAGE FROM CLI
# if __name__ == "__main__":
#     target_fun_str, domain, degree, continuity = parseCommandLineArguments( )
#     target_fun, spline_space = prepareCommandInputs( target_fun_str, domain, degree, continuity )
#     main( target_fun, spline_space )


class Test_assembleGramMatrix( unittest.TestCase ):
    def test_two_element_linear_bspline( self ):
        target_fun = lambda x: x**0
        spline_space = { "domain": [0, 2], "degree": [ 1, 1 ], "continuity": [ -1, 0, -1 ] }
        continuity = spline_space["continuity"]
        uspline.make_uspline_mesh( spline_space, "temp_uspline" )
        uspline_bext = bext.readBEXT( "temp_uspline.json" )
        test_gram_matrix = assembleGramMatrix(continuity, uspline_bext = uspline_bext )
        gold_gram_matrix = numpy.array( [ [ 1.0/3.0, 1.0/6.0, 0.0 ],
                                          [ 1.0/6.0, 2.0/3.0, 1.0/6.0 ],
                                          [ 0.0, 1.0/6.0, 1.0/3.0 ] ] )
        self.assertTrue( numpy.allclose( test_gram_matrix, gold_gram_matrix ) )

    def test_two_element_quadratic_bspline( self ):
        target_fun = lambda x: x**0
        spline_space = { "domain": [0, 2], "degree": [ 2, 2 ], "continuity": [ -1, 1, -1 ] }
        continuity = spline_space["continuity"]
        uspline.make_uspline_mesh( spline_space, "temp_uspline" )
        uspline_bext = bext.readBEXT( "temp_uspline.json" )
        test_gram_matrix = assembleGramMatrix(continuity, uspline_bext = uspline_bext )
        gold_gram_matrix = numpy.array( [ [ 1.0/5.0, 7.0/60.0, 1.0/60.0, 0.0 ],
                                          [ 7.0/60.0, 1.0/3.0, 1.0/5.0, 1.0/60.0],
                                          [ 1.0/60.0, 1.0/5.0, 1.0/3.0, 7.0/60.0 ],
                                          [ 0.0, 1.0/60.0, 7.0/60.0, 1.0/5.0] ] )
        self.assertTrue( numpy.allclose( test_gram_matrix, gold_gram_matrix ) )

    def test_two_element_cubic_bspline( self ):
        spline_space = { "domain": [0, 2], "degree": [ 3, 3 ], "continuity": [ -1, 2, -1 ] }
        continuity = spline_space["continuity"]
        uspline.make_uspline_mesh( spline_space, "temp_uspline" )
        uspline_bext = bext.readBEXT( "temp_uspline.json" )
        test_gram_matrix = assembleGramMatrix( continuity, uspline_bext = uspline_bext )
        gold_gram_matrix = numpy.array( [ [ 1.0/7.0, 7.0/80.0, 1.0/56.0, 1.0/560.0, 0.0 ],
                                          [ 7.0/80.0, 31.0/140.0, 39.0/280.0, 1.0/20.0, 1.0/560.0 ],
                                          [ 1.0/56.0, 39.0/280.0, 13.0/70.0, 39.0/280.0, 1.0/56.0 ],
                                          [ 1.0/560.0, 1.0/20.0, 39.0/280.0, 31.0/140.0, 7.0/80.0 ],
                                          [ 0.0, 1.0/560.0, 1.0/56.0, 7.0/80.0, 1.0/7.0 ] ] )
        self.assertTrue( numpy.allclose( test_gram_matrix, gold_gram_matrix ) )
        
# class Test_assembleForceVector( unittest.TestCase ):
#     def test_const_force_fun_two_element_linear_bspline( self ):
#         target_fun = lambda x: numpy.pi
#         spline_space = { "domain": [-1, 1], "degree": [ 1, 1 ], "continuity": [ -1, 0, -1 ] }
#         continuity = spline_space["continuity"]
#         uspline.make_uspline_mesh( spline_space, "temp_uspline" )
#         uspline_bext = bext.readBEXT( "temp_uspline.json" )
#         test_force_vector = assembleForceVector( target_fun = target_fun, uspline_bext = uspline_bext )
#         gold_force_vector = numpy.array( [ numpy.pi / 2.0, numpy.pi, numpy.pi / 2.0 ] )
#         self.assertTrue( numpy.allclose( test_force_vector, gold_force_vector ) )

#     def test_linear_force_fun_two_element_linear_bspline( self ):
#         target_fun = lambda x: x
#         spline_space = { "domain": [-1, 1], "degree": [ 1, 1 ], "continuity": [ -1, 0, -1 ] }
#         continuity = spline_space["continuity"]
#         uspline.make_uspline_mesh( spline_space, "temp_uspline" )
#         uspline_bext = bext.readBEXT( "temp_uspline.json" )
#         test_force_vector = assembleForceVector( target_fun = target_fun, uspline_bext = uspline_bext )
#         gold_force_vector = numpy.array( [ -1.0/3.0, 0.0, 1.0/3.0 ] )
#         self.assertTrue( numpy.allclose( test_force_vector, gold_force_vector ) )

#     def test_quadratic_force_fun_two_element_linear_bspline( self ):
#         target_fun = lambda x: x**2
#         spline_space = { "domain": [-1, 1], "degree": [ 1, 1 ], "continuity": [ -1, 0, -1 ] }
#         continuity = spline_space["continuity"]
#         uspline.make_uspline_mesh( spline_space, "temp_uspline" )
#         uspline_bext = bext.readBEXT( "temp_uspline.json" )
#         test_force_vector = assembleForceVector( target_fun = target_fun, uspline_bext = uspline_bext )
#         gold_force_vector = numpy.array( [ 1.0/4.0, 1.0/6.0, 1.0/4.0 ] )
#         self.assertTrue( numpy.allclose( test_force_vector, gold_force_vector ) )

#     def test_const_force_fun_two_element_quadratic_bspline( self ):
#         target_fun = lambda x: numpy.pi
#         spline_space = { "domain": [-1, 1], "degree": [ 2, 2 ], "continuity": [ -1, 1, -1 ] }
#         continuity = spline_space["continuity"]
#         uspline.make_uspline_mesh( spline_space, "temp_uspline" )
#         uspline_bext = bext.readBEXT( "temp_uspline.json" )
#         test_force_vector = assembleForceVector( target_fun = target_fun, uspline_bext = uspline_bext )
#         gold_force_vector = numpy.array( [ numpy.pi/3.0, 2.0*numpy.pi/3.0, 2.0*numpy.pi/3.0, numpy.pi/3.0 ] )
#         self.assertTrue( numpy.allclose( test_force_vector, gold_force_vector ) )

#     def test_linear_force_fun_two_element_quadratic_bspline( self ):
#         target_fun = lambda x: x
#         spline_space = { "domain": [-1, 1], "degree": [ 2, 2 ], "continuity": [ -1, 1, -1 ] }
#         continuity = spline_space["continuity"]
#         uspline.make_uspline_mesh( spline_space, "temp_uspline" )
#         uspline_bext = bext.readBEXT( "temp_uspline.json" )
#         test_force_vector = assembleForceVector( target_fun = target_fun, uspline_bext = uspline_bext )
#         gold_force_vector = numpy.array( [ -1.0/4.0, -1.0/6.0, 1.0/6.0, 1.0/4.0 ] )
#         self.assertTrue( numpy.allclose( test_force_vector, gold_force_vector ) )

#     def test_quadratic_force_fun_two_element_quadratic_bspline( self ):
#         target_fun = lambda x: x**2
#         spline_space = { "domain": [-1, 1], "degree": [ 2, 2 ], "continuity": [ -1, 1, -1 ] }
#         continuity = spline_space["continuity"]
#         uspline.make_uspline_mesh( spline_space, "temp_uspline" )
#         uspline_bext = bext.readBEXT( "temp_uspline.json" )
#         test_force_vector = assembleForceVector( target_fun = target_fun, uspline_bext = uspline_bext )
#         gold_force_vector = numpy.array( [ 2.0/10.0, 2.0/15.0, 2.0/15.0, 2.0/10.0 ] )
#         self.assertTrue( numpy.allclose( test_force_vector, gold_force_vector ) )
        
# class Test_computeSolution( unittest.TestCase ):
#     def test_cubic_polynomial_target_linear_bspline( self ):
#         # print( "POLY TEST" )
#         target_fun = lambda x: x**3 - (8/5)*x**2 + (3/5)*x
#         spline_space = { "domain": [0, 1], "degree": [ 1, 1 ], "continuity": [ -1, 0, -1 ] }
#         uspline.make_uspline_mesh( spline_space, "temp_uspline" )
#         uspline_bext = bext.readBEXT( "temp_uspline.json" )
#         test_sol_coeff = computeSolution( target_fun = target_fun, uspline_bext = uspline_bext )
#         # plotCompareFunToTestSolution( target_fun, test_sol_coeff, uspline_bext )
#         gold_sol_coeff = numpy.array( [ 9.0/160.0, 7.0/240.0, -23.0/480.0 ] )
#         abs_err, rel_err = computeFitError( target_fun, test_sol_coeff, uspline_bext )
#         self.assertTrue( numpy.allclose( gold_sol_coeff, test_sol_coeff ) )
#         self.assertAlmostEqual( first = rel_err, second = 0, delta = 1e0 )

#     def test_cubic_polynomial_target_quadratic_bspline( self ):
#         # print( "POLY TEST" )
#         target_fun = lambda x: x**3 - (8/5)*x**2 + (3/5)*x
#         spline_space = { "domain": [0, 1], "degree": [ 2, 2 ], "continuity": [ -1, 1, -1 ] }
#         uspline.make_uspline_mesh( spline_space, "temp_uspline" )
#         uspline_bext = bext.readBEXT( "temp_uspline.json" )
#         test_sol_coeff = computeSolution( target_fun = target_fun, uspline_bext = uspline_bext )
#         # plotCompareFunToTestSolution( target_fun, test_sol_coeff, uspline_bext )
#         gold_sol_coeff = numpy.array( [ 1.0/120.0, 9.0/80.0, -1.0/16.0, -1.0/120.0 ] )
#         abs_err, rel_err = computeFitError( target_fun, test_sol_coeff, uspline_bext )
#         self.assertTrue( numpy.allclose( gold_sol_coeff, test_sol_coeff ) )
#         self.assertAlmostEqual( first = rel_err, second = 0, delta = 1e-1 )

#     def test_cubic_polynomial_target_cubic_bspline( self ):
#         # print( "POLY TEST" )
#         target_fun = lambda x: x**3 - (8/5)*x**2 + (3/5)*x
#         spline_space = { "domain": [0, 1], "degree": [ 3, 3 ], "continuity": [ -1, 2, -1 ] }
#         uspline.make_uspline_mesh( spline_space, "temp_uspline" )
#         uspline_bext = bext.readBEXT( "temp_uspline.json" )
#         test_sol_coeff = computeSolution( target_fun = target_fun, uspline_bext = uspline_bext )
#         # plotCompareFunToTestSolution( target_fun, test_sol_coeff, uspline_bext )
#         gold_sol_coeff = numpy.array( [ 0.0, 1.0/10.0, 1.0/30.0, -1.0/15.0, 0.0 ] )
#         abs_err, rel_err = computeFitError( target_fun, test_sol_coeff, uspline_bext )
#         self.assertTrue( numpy.allclose( gold_sol_coeff, test_sol_coeff ) )
#         self.assertAlmostEqual( first = rel_err, second = 0, delta = 1e-12 )

#     def test_sin_target( self ):
#         # print( "SIN TEST" )
#         target_fun = lambda x: numpy.sin( numpy.pi * x )
#         spline_space = { "domain": [0, 1], "degree": [ 3, 3 ], "continuity": [ -1, 2, -1 ] }
#         uspline.make_uspline_mesh( spline_space, "temp_uspline" )
#         uspline_bext = bext.readBEXT( "temp_uspline.json" )
#         test_sol_coeff = computeSolution( target_fun = target_fun, uspline_bext = uspline_bext )
#         abs_err, rel_err = computeFitError( target_fun, test_sol_coeff, uspline_bext )
#         # plotCompareFunToTestSolution( target_fun, test_sol_coeff, uspline_bext )
#         self.assertAlmostEqual( first = rel_err, second = 0, delta = 1e-2 )

#     def test_erfc_target( self ):
#         # print( "ERFC TEST" )
#         target_fun = lambda x: numpy.real( scipy.special.erfc( x ) )
#         spline_space = { "domain": [-1, 1], "degree": [ 3, 1, 3 ], "continuity": [ -1, 1, 1, -1 ] }
#         uspline.make_uspline_mesh( spline_space, "temp_uspline" )
#         uspline_bext = bext.readBEXT( "temp_uspline.json" )
#         test_sol_coeff = computeSolution( target_fun = target_fun, uspline_bext = uspline_bext )
#         abs_err, rel_err = computeFitError( target_fun, test_sol_coeff, uspline_bext )
#         # plotCompareFunToTestSolution( target_fun, test_sol_coeff, uspline_bext )
#         self.assertAlmostEqual( first = rel_err, second = 0, delta = 1e-2 )

#     def test_exptx_target( self ):
#         # print( "EXPT TEST" )
#         target_fun = lambda x: float( numpy.real( float( x )**float( x ) ) )
#         spline_space = { "domain": [-1, 1], "degree": [ 5, 5, 5, 5 ], "continuity": [ -1, 4, 0, 4, -1 ] }
#         uspline.make_uspline_mesh( spline_space, "temp_uspline" )
#         uspline_bext = bext.readBEXT( "temp_uspline.json" )
#         test_sol_coeff = computeSolution( target_fun = target_fun, uspline_bext = uspline_bext )
#         abs_err, rel_err = computeFitError( target_fun, test_sol_coeff, uspline_bext )
#         # plotCompareFunToTestSolution( target_fun, test_sol_coeff, uspline_bext )
#         self.assertAlmostEqual( first = rel_err, second = 0, delta = 1e-2 )
        
unittest.main()