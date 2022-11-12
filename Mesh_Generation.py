# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 10:31:33 2022

@author: rphar
"""
import numpy
import sympy
import unittest

# import Basis_Func as BF

def generateMesh(xmin,xmax,degree):
    num_elems = len(degree)
    node_coords = []
    ien_array = []
    elem_bound = numpy.linspace(xmin, xmax, num_elems + 1)
    
    counter0 = int(0)
    elem_ien = []
    for i in range(0,degree[0]+1):
        elem_ien.append(counter0)
        counter0 += int(1)
    ien_array.append(elem_ien)
    
    for e in range(0,num_elems):
        elem_xmin = elem_bound[e]
        elem_xmax = elem_bound[e+1]
        
        elem_nodes = numpy.linspace(elem_xmin,elem_xmax,degree[e]+1)
        node_coords.append(elem_nodes)
    
       
        
        #following entries to ien array
        if e == 0:
            continue
        else: 
            elem_ien = []
            ien_array.append(elem_ien)
            counter1 = ien_array[-2][-1]
            for i in range(0,degree[e]+1):
                ien_array[-1].append(counter1)
                counter1 += 1
    
    
    return node_coords, ien_array

def generateMesh1D(xmin,xmax,num_elems,degree):
    node_coords = numpy.linspace(xmin,xmax,int(degree * num_elems+1))

    ien_array = numpy.zeros((num_elems,degree+1))
    for i in range(0,num_elems):
        local_elem = []
        for j in range(0,degree+1):
            local_elem.append(i*degree+j)
        ien_array[i,:] = local_elem
    
    return node_coords, ien_array

def computeSolution(target_fun,domain,num_elems,degree):
    xmin = domain[0]
    xmax = domain[1]
    node_coords ,ien_array = generateMesh1D(xmin, xmax, num_elems, degree)
    sol_coeffs = target_fun(node_coords)
    return sol_coeffs

def getElementIdxContainingPoint(x,node_coords,ien_array):
    num_elems = len(ien_array)
    for elem_idx in range(0,num_elems):
        elem_boundary_node_ids = [ien_array[elem_idx][0],ien_array[elem_idx][-1]]
        elem_boundary_coords = [node_coords[elem_boundary_node_ids[0]],node_coords[elem_boundary_node_ids[1]]]
        if x >= elem_boundary_coords[0] and x <= elem_boundary_coords[1]:        
            return elem_idx

def getElementNodes(ien_array,elem_idx):
    Elem_nodes = ien_array[elem_idx]
    return Elem_nodes

def getElementDomain(node_coords,ien_array,elem_idx):
    domain = [0,0]
    domain[0] = node_coords[elem_idx][0] 
    domain[1] = node_coords[elem_idx][-1]
    return domain

def spatialToParamCoords(x,elem_domain):
    xmin = elem_domain[0]
    xmax = elem_domain[1]
    length = xmax - xmin
    
    param_x = ((x - xmin)/(length))*2
    return param_x

def basisEval(degree,elem_idx,param_coord):
    return

def evaluateSolutionAt( x, coeff, node_coords, ien_array, eval_basis):
    elem_idx = getElementIdxContainingPoint(x, node_coords, ien_array)
    elem_nodes = getElementNodes(ien_array, elem_idx)
    elem_domain = getElementDomain(node_coords,ien_array, elem_idx)
    param_coord = spatialToParamCoords(x, elem_domain)
    degree = len(ien_array[0]-1)
    sol_at_point = 0
    for i in range(0,len(elem_nodes)):
        curr_node = elem_nodes[i]
        sol_at_point += coeff[curr_node] * basisEval(degree, i, param_coord)
    return sol_at_point

class Test_generateMesh1D( unittest.TestCase ):
    def test_make_1_linear_elem( self ):
        gold_node_coords = numpy.array( [ 0.0, 1.0 ] )
        gold_ien_array = numpy.array( [ [ 0, 1 ] ], dtype = int )
        node_coords, ien_array = generateMesh1D( xmin = 0.0, xmax = 1.0, num_elems = 1, degree = 1 )
        self.assertIsInstance( obj = node_coords, cls = numpy.ndarray )
        self.assertIsInstance( obj = ien_array, cls = numpy.ndarray )
        self.assertTrue( numpy.allclose( node_coords, gold_node_coords ) )
        self.assertTrue( numpy.array_equiv( ien_array, gold_ien_array ) )
    
    def test_make_1_quadratic_elem( self ):
        gold_node_coords = numpy.array( [ 0.0, 0.5, 1.0 ] )
        gold_ien_array = numpy.array( [ [ 0, 1, 2 ] ], dtype = int )
        node_coords, ien_array = generateMesh1D( xmin = 0.0, xmax = 1.0, num_elems = 1, degree = 2 )
        self.assertIsInstance( obj = node_coords, cls = numpy.ndarray )
        self.assertIsInstance( obj = ien_array, cls = numpy.ndarray )
        self.assertTrue( numpy.allclose( node_coords, gold_node_coords ) )
        self.assertTrue( numpy.array_equiv( ien_array, gold_ien_array ) )
    
    def test_make_2_linear_elems( self ):
        gold_node_coords = numpy.array( [ 0.0, 0.5, 1.0 ] )
        gold_ien_array = numpy.array( [ [ 0, 1 ], [ 1, 2 ] ], dtype = int )
        node_coords, ien_array = generateMesh1D( xmin = 0.0, xmax = 1.0, num_elems = 2, degree = 1 )
        self.assertIsInstance( obj = node_coords, cls = numpy.ndarray )
        self.assertIsInstance( obj = ien_array, cls = numpy.ndarray )
        self.assertTrue( numpy.allclose( node_coords, gold_node_coords ) )
        self.assertTrue( numpy.array_equiv( ien_array, gold_ien_array ) )
        
    def test_make_2_quadratic_elems( self ):
        gold_node_coords = numpy.array( [ 0.0, 0.25, 0.5, 0.75, 1.0 ] )
        gold_ien_array = numpy.array( [ [ 0, 1, 2 ], [ 2, 3, 4 ] ], dtype = int )
        node_coords, ien_array = generateMesh1D( xmin = 0.0, xmax = 1.0, num_elems = 2, degree = 2 )
        self.assertIsInstance( obj = node_coords, cls = numpy.ndarray )
        self.assertIsInstance( obj = ien_array, cls = numpy.ndarray )
        self.assertTrue( numpy.allclose( node_coords, gold_node_coords ) )
        self.assertTrue( numpy.array_equiv( ien_array, gold_ien_array ) )
    
    def test_make_4_linear_elems( self ):
        gold_node_coords = numpy.array( [ 0.0, 0.25, 0.5, 0.75, 1.0 ] )
        gold_ien_array = numpy.array( [ [ 0, 1 ], [ 1, 2 ], [ 2, 3 ], [ 3, 4 ] ], dtype = int )
        node_coords, ien_array = generateMesh1D( xmin = 0.0, xmax = 1.0, num_elems = 4, degree = 1 )
        self.assertIsInstance( obj = node_coords, cls = numpy.ndarray )
        self.assertIsInstance( obj = ien_array, cls = numpy.ndarray )
        self.assertTrue( numpy.allclose( node_coords, gold_node_coords ) )
        self.assertTrue( numpy.array_equiv( ien_array, gold_ien_array ) )
    
    def test_make_4_quadratic_elems( self ):
        gold_node_coords = numpy.array( [ 0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0 ] )
        gold_ien_array = numpy.array( [ [ 0, 1, 2 ], [ 2, 3, 4 ], [ 4, 5, 6 ], [ 6, 7, 8 ] ], dtype = int )
        node_coords, ien_array = generateMesh1D( xmin = 0.0, xmax = 1.0, num_elems = 4, degree = 2 )
        self.assertIsInstance( obj = node_coords, cls = numpy.ndarray )
        self.assertIsInstance( obj = ien_array, cls = numpy.ndarray )
        self.assertTrue( numpy.allclose( node_coords, gold_node_coords ) )
        self.assertTrue( numpy.array_equiv( ien_array, gold_ien_array ) )
        
# class Test_computeSolution( unittest.TestCase ):
#     def test_single_linear_element_poly( self ):
#         test_solution, node_coords, ien_array = computeSolution( target_fun = lambda x : x, domain = [-1.0, 1.0 ], num_elems = 1, degree = 1 )
#         gold_solution = numpy.array( [ -1.0, 1.0 ] )
#         self.assertTrue( numpy.allclose( test_solution, gold_solution ) )

#     def test_single_quad_element_poly( self ):
#         test_solution, node_coords, ien_array = computeSolution( target_fun = lambda x : x**2, domain = [-1.0, 1.0 ], num_elems = 1, degree = 2 )
#         gold_solution = numpy.array( [ 1.0, 0.0, 1.0 ] )
#         self.assertTrue( numpy.allclose( test_solution, gold_solution ) )

#     def test_two_linear_element_poly( self ):
#         test_solution, node_coords, ien_array = computeSolution( target_fun = lambda x : x**2, domain = [-1.0, 1.0 ], num_elems = 2, degree = 1 )
#         gold_solution = numpy.array( [ 1.0, 0.0, 1.0 ] )
#         self.assertTrue( numpy.allclose( test_solution, gold_solution ) )

#     def test_four_quad_element_poly( self ):
#         test_solution, node_coords, ien_array = computeSolution( target_fun = lambda x : x**2, domain = [-1.0, 1.0 ], num_elems = 4, degree = 1 )
#         gold_solution = numpy.array( [ 1.0, 0.25, 0.0, 0.25, 1.0 ] )
#         self.assertTrue( numpy.allclose( test_solution, gold_solution ) )
        
# class Test_evaluateSolutionAt( unittest.TestCase ):
#     def test_single_linear_element( self ):
#         node_coords, ien_array = mesh.generateMesh( -1, 1, 1, 1 )
#         coeff = numpy.array( [-1.0, 1.0 ] )
#         self.assertAlmostEqual( first = evaluateSolutionAt( x = -1.0, coeff = coeff, node_coords = node_coords, ien_array = ien_array, eval_basis = basis.evalLagrangeBasis1D ), second = -1.0 )
#         self.assertAlmostEqual( first = evaluateSolutionAt( x =  0.0, coeff = coeff, node_coords = node_coords, ien_array = ien_array, eval_basis = basis.evalLagrangeBasis1D ), second =  0.0 )
#         self.assertAlmostEqual( first = evaluateSolutionAt( x = +1.0, coeff = coeff, node_coords = node_coords, ien_array = ien_array, eval_basis = basis.evalLagrangeBasis1D ), second = +1.0 )
        
#     def test_two_linear_elements( self ):
#         node_coords, ien_array = mesh.generateMesh( -1, 1, 2, 1 )
#         coeff = numpy.array( [ 1.0, 0.0, 1.0 ] )
#         self.assertAlmostEqual( first = evaluateSolutionAt( x = -1.0, coeff = coeff, node_coords = node_coords, ien_array = ien_array, eval_basis = basis.evalLagrangeBasis1D ), second = +1.0 )
#         self.assertAlmostEqual( first = evaluateSolutionAt( x =  0.0, coeff = coeff, node_coords = node_coords, ien_array = ien_array, eval_basis = basis.evalLagrangeBasis1D ), second =  0.0 )
#         self.assertAlmostEqual( first = evaluateSolutionAt( x = +1.0, coeff = coeff, node_coords = node_coords, ien_array = ien_array, eval_basis = basis.evalLagrangeBasis1D ), second = +1.0 )
        
#     def test_single_quadratic_element( self ):
#         node_coords, ien_array = mesh.generateMesh( -1, 1, 1, 2 )
#         coeff = numpy.array( [+1.0, 0.0, 1.0 ] )
#         self.assertAlmostEqual( first = evaluateSolutionAt( x = -1.0, coeff = coeff, node_coords = node_coords, ien_array = ien_array, eval_basis = basis.evalLagrangeBasis1D ), second = +1.0 )
#         self.assertAlmostEqual( first = evaluateSolutionAt( x =  0.0, coeff = coeff, node_coords = node_coords, ien_array = ien_array, eval_basis = basis.evalLagrangeBasis1D ), second =  0.0 )
#         self.assertAlmostEqual( first = evaluateSolutionAt( x = +1.0, coeff = coeff, node_coords = node_coords, ien_array = ien_array, eval_basis = basis.evalLagrangeBasis1D ), second = +1.0 )
        
#     def test_two_quadratic_elements( self ):
#         node_coords, ien_array = mesh.generateMesh( -2, 2, 2, 2 )
#         coeff = numpy.array( [ 1.0, 0.25, 0.5, 0.25, 1.0 ] )
#         self.assertAlmostEqual( first = evaluateSolutionAt( x = -2.0, coeff = coeff, node_coords = node_coords, ien_array = ien_array, eval_basis = basis.evalLagrangeBasis1D ), second = +1.00 )
#         self.assertAlmostEqual( first = evaluateSolutionAt( x = -1.0, coeff = coeff, node_coords = node_coords, ien_array = ien_array, eval_basis = basis.evalLagrangeBasis1D ), second = +0.25 )
#         self.assertAlmostEqual( first = evaluateSolutionAt( x =  0.0, coeff = coeff, node_coords = node_coords, ien_array = ien_array, eval_basis = basis.evalLagrangeBasis1D ), second = +0.50 )
#         self.assertAlmostEqual( first = evaluateSolutionAt( x = +1.0, coeff = coeff, node_coords = node_coords, ien_array = ien_array, eval_basis = basis.evalLagrangeBasis1D ), second = +0.25 )
#         self.assertAlmostEqual( first = evaluateSolutionAt( x = +2.0, coeff = coeff, node_coords = node_coords, ien_array = ien_array, eval_basis = basis.evalLagrangeBasis1D ), second = +1.00 )

# unittest.main()