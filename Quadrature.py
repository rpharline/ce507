# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 14:13:35 2022

@author: rphar
"""
import sys
import math
import scipy
from scipy import optimize
import numpy
import unittest
import sympy

import Basis_Func as BF
# from src import basis

#creates midpoint reimann Quad along domain [-1,1]
def riemannQuadrature(fun, num_points):
    x,w = getRiemannQuadrature(num_points)
    f_x = []
    for value in x:
        f_x.append(fun(value))
    
    total = 0
    for i in range(0,len(f_x)):
        total += f_x[i]*w[i]
    
    return total
    
#gets points to be plugged in for Riemann Quad and weights along domain [-1,1]
def getRiemannQuadrature(num_points):
    if num_points <= 0:
        raise Exception("num_points_MUST_BE_INTEGER_GEQ_1")
        
    
    else:
        nodes = numpy.linspace(-1,1,num_points+1)
        
        x = numpy.zeros(num_points)
        w = numpy.zeros(num_points)
        for i in range(0,num_points):
            x[i] = (nodes[i]+nodes[i+1])/2
            w[i] = nodes[i+1]-nodes[i]
    
    return x,w

#gets points and weights for Newton-Cotes Quadrature
def getNewtonCotesQuadrature(num_points):
    if num_points == 0 or num_points > 6:
        raise Exception("num_points_MUST_BE_INTEGER_IN_[1,6]")
    elif num_points == 1:
        x = numpy.array([0])
    else:
        x = numpy.linspace(-1,1,num_points)
        
    w = numpy.zeros(num_points)
    for i in range(0,num_points):
        var = sympy.Symbol('x')
        integral = sympy.integrate(BF.evaluateLagrangeBasis1D(var, num_points-1, i),(var,-1,1))
        w[i] = (integral)
    
    return x,w

#Performs Newton-Cotes Quadrature on a function along domain [-1,1]
def computeNewtonCotesQuadrature(fun,num_points):
    x,w  = getNewtonCotesQuadrature(num_points)
    coeff = []
    point_idx = 0
    for i in x:
        coeff.append(fun(x[point_idx]))
        point_idx += 1
      
    total = 0
    for i in range(0,num_points):
        total += w[i]*coeff[i]
    
    return total

def computeGaussLegendreQuadrature( n ):
    M = numpy.zeros( 2*n, dtype = "double" )
    M[0] = 2.0
    x0 = numpy.linspace( -1, 1, n )
    sol = scipy.optimize.least_squares( lambda x : objFun( M, x ), x0, bounds = (-1, 1), ftol = 1e-14, xtol = 1e-14, gtol = 1e-14 )
    qp = sol.x
    w = solveLinearMomentFit( M, qp )
    return qp, w

def assembleLinearMomentFitSystem( degree, pts ):
    A = numpy.zeros( shape = ( degree + 1, len( pts ) ), dtype = "double" )
    for i in range(0,degree+1):
        for j in range(0,len(pts)):
            A[i,j] = BF.evalLegendreBasis1D(i, pts[j])
    
    return A

def solveLinearMomentFit( M, pts ):
    degree = len( M ) - 1
    A = assembleLinearMomentFitSystem( degree, pts )
    sol = scipy.optimize.lsq_linear( A, M )
    w = sol.x
    return w

def objFun( M, pts ):
    degree = len( M ) - 1
    A = assembleLinearMomentFitSystem( degree, pts )
    w = solveLinearMomentFit( M, pts )
    
    obj_val = numpy.squeeze(M - numpy.matmul(A,w))
    
    return obj_val


class Test_computeRiemannQuadrature( unittest.TestCase ):
    def test_integrate_constant_one( self ):
        constant_one = lambda x : 1
        for num_points in range( 1, 100 ):
            self.assertAlmostEqual( first = riemannQuadrature( fun = constant_one, num_points = num_points ), second = 2.0, delta = 1e-12 )

    def test_integrate_linear( self ):
        linear = lambda x : x
        for num_points in range( 1, 100 ):
            self.assertAlmostEqual( first = riemannQuadrature( fun = linear, num_points = num_points ), second = 0.0, delta = 1e-12 )

    def test_integrate_quadratic( self ):
        linear = lambda x : x**2
        error = []
        for num_points in range( 1, 100 ):
            error.append( abs( (2.0 / 3.0) - riemannQuadrature( fun = linear, num_points = num_points ) ) )
        self.assertTrue( numpy.all( numpy.diff( error ) <= 0.0 ) )

    def test_integrate_sin( self ):
        sin = lambda x : math.sin(x)
        error = []
        for num_points in range( 1, 100 ):
            self.assertAlmostEqual( first = riemannQuadrature( fun = sin, num_points = num_points ), second = 0.0, delta = 1e-12 )

    def test_integrate_cos( self ):
        cos = lambda x : math.cos(x)
        error = []
        for num_points in range( 1, 100 ):
            error.append( abs( (2.0 / 3.0) - riemannQuadrature( fun = cos, num_points = num_points ) ) )
        self.assertTrue( numpy.all( numpy.diff( error ) <= 0.0 ) )

class Test_getRiemannQuadrature( unittest.TestCase ):
    def test_zero_points( self ):
        with self.assertRaises( Exception ) as context:
            getRiemannQuadrature( num_points = 0 )
        self.assertEqual( "num_points_MUST_BE_INTEGER_GEQ_1", str( context.exception ) )

    def test_one_point( self ):
        x, w = getRiemannQuadrature( num_points = 1 )
        self.assertAlmostEqual( first = x, second = 0.0 )
        self.assertAlmostEqual( first = w, second = 2.0 )
        self.assertIsInstance( obj = x, cls = numpy.ndarray )
        self.assertIsInstance( obj = w, cls = numpy.ndarray )

    def test_two_point( self ):
        x, w = getRiemannQuadrature( num_points = 2 )
        self.assertTrue( numpy.allclose( x, [ -0.50, 0.50 ] ) )
        self.assertTrue( numpy.allclose( w, [ 1.0, 1.0 ] ) )
        self.assertIsInstance( obj = x, cls = numpy.ndarray )
        self.assertIsInstance( obj = w, cls = numpy.ndarray )

    def test_three_point( self ):
        x, w = getRiemannQuadrature( num_points = 3 )
        self.assertTrue( numpy.allclose( x, [ -2.0/3.0, 0.0, 2.0/3.0 ] ) )
        self.assertTrue( numpy.allclose( w, [ 2.0/3.0, 2.0/3.0, 2.0/3.0 ] ) )
        self.assertIsInstance( obj = x, cls = numpy.ndarray )
        self.assertIsInstance( obj = w, cls = numpy.ndarray )

    def test_many_points( self ):
        for num_points in range( 1, 100 ):
            x, w = getRiemannQuadrature( num_points = num_points )
            self.assertTrue( len( x ) == num_points )
            self.assertTrue( len( w ) == num_points )
            self.assertIsInstance( obj = x, cls = numpy.ndarray )
            self.assertIsInstance( obj = w, cls = numpy.ndarray )

class Test_computeNewtonCotesQuadrature( unittest.TestCase ):
    def test_integrate_constant_one( self ):
        constant_one = lambda x : 1 * x**0
        for degree in range( 1, 6 ):
            num_points = degree + 1
            self.assertAlmostEqual( first = computeNewtonCotesQuadrature( fun = constant_one, num_points = num_points ), second = 2.0, delta = 1e-12 )

    def test_exact_poly_int( self ):
        for degree in range( 1, 6 ):
            num_points = degree + 1
            poly_fun = lambda x : ( x + 1.0 ) ** degree
            indef_int = lambda x : ( ( x + 1 ) ** ( degree + 1) ) / ( degree + 1 )
            def_int = indef_int(1.0) - indef_int(-1.0)
            self.assertAlmostEqual( first = computeNewtonCotesQuadrature( fun = poly_fun, num_points = num_points ), second = def_int, delta = 1e-12 )

    def test_integrate_sin( self ):
        sin = lambda x : math.sin(x)
        for num_points in range( 1, 7 ):
            self.assertAlmostEqual( first = computeNewtonCotesQuadrature( fun = sin, num_points = num_points ), second = 0.0, delta = 1e-12 )

    def test_integrate_cos( self ):
        cos = lambda x : math.cos(x)
        self.assertAlmostEqual( first = computeNewtonCotesQuadrature( fun = cos, num_points = 6 ), second = 2*math.sin(1), delta = 1e-4 )

class Test_getNewtonCotesQuadrature( unittest.TestCase ):
    def test_incorrect_num_points( self ):
        with self.assertRaises( Exception ) as context:
            getNewtonCotesQuadrature( num_points = 0 )
        self.assertEqual( "num_points_MUST_BE_INTEGER_IN_[1,6]", str( context.exception ) )
        with self.assertRaises( Exception ) as context:
            getNewtonCotesQuadrature( num_points = 7 )
        self.assertEqual( "num_points_MUST_BE_INTEGER_IN_[1,6]", str( context.exception ) )

    def test_return_types( self ):
        for num_points in range( 1, 7 ):
            x, w = getNewtonCotesQuadrature( num_points = num_points )
            self.assertIsInstance( obj = x, cls = numpy.ndarray )
            self.assertIsInstance( obj = w, cls = numpy.ndarray )
            self.assertTrue( len( x ) == num_points )
            self.assertTrue( len( w ) == num_points )

class Test_computeGaussLegendreQuadrature( unittest.TestCase ):
    def test_1_pt( self ):
        qp_gold = numpy.array( [ 0.0 ] )
        w_gold = numpy.array( [ 2.0 ] )
        [ qp, w ] = computeGaussLegendreQuadrature( 1 )
        self.assertAlmostEqual( first = qp, second = qp_gold, delta = 1e-12 )
        self.assertAlmostEqual( first = w, second = w_gold, delta = 1e-12 )

    def test_2_pt( self ):
        qp_gold = numpy.array( [ -1.0/numpy.sqrt(3), 1.0/numpy.sqrt(3) ] )
        w_gold = numpy.array( [ 1.0, 1.0 ] )
        [ qp, w ] = computeGaussLegendreQuadrature( 2 )
        self.assertTrue( numpy.allclose( qp, qp_gold ) )
        self.assertTrue( numpy.allclose( w, w_gold ) )

    def test_3_pt( self ):
        qp_gold = numpy.array( [ -1.0 * numpy.sqrt( 3.0 / 5.0 ),
                                0.0,
                                +1.0 * numpy.sqrt( 3.0 / 5.0 ) ] )
        w_gold = numpy.array( [ 5.0 / 9.0,
                                8.0 / 9.0,
                                5.0 / 9.0 ] )
        [ qp, w ] = computeGaussLegendreQuadrature( 3 )
        self.assertTrue( numpy.allclose( qp, qp_gold ) )
        self.assertTrue( numpy.allclose( w, w_gold ) )

    def test_4_pt( self ):
        qp_gold = numpy.array( [ -1.0 * numpy.sqrt( 3.0 / 7.0 + 2.0 / 7.0 * numpy.sqrt( 6.0 / 5.0 ) ),
                                -1.0 * numpy.sqrt( 3.0 / 7.0 - 2.0 / 7.0 * numpy.sqrt( 6.0 / 5.0 ) ),
                                +1.0 * numpy.sqrt( 3.0 / 7.0 - 2.0 / 7.0 * numpy.sqrt( 6.0 / 5.0 ) ),
                                +1.0 * numpy.sqrt( 3.0 / 7.0 + 2.0 / 7.0 * numpy.sqrt( 6.0 / 5.0 ) ) ] )
        w_gold = numpy.array( [ ( 18.0 - numpy.sqrt( 30.0 ) ) / 36.0,
                                ( 18.0 + numpy.sqrt( 30.0 ) ) / 36.0,
                                ( 18.0 + numpy.sqrt( 30.0 ) ) / 36.0,
                                ( 18.0 - numpy.sqrt( 30.0 ) ) / 36.0 ] )
        [ qp, w ] = computeGaussLegendreQuadrature( 4 )
        self.assertTrue( numpy.allclose( qp, qp_gold ) )
        self.assertTrue( numpy.allclose( w, w_gold ) )

    def test_5_pt( self ):
        qp_gold = numpy.array( [ -1.0 / 3.0 * numpy.sqrt( 5.0 + 2.0 * numpy.sqrt( 10.0 / 7.0 ) ),
                                -1.0 / 3.0 * numpy.sqrt( 5.0 - 2.0 * numpy.sqrt( 10.0 / 7.0 ) ),
                                0.0,
                                +1.0 / 3.0 * numpy.sqrt( 5.0 - 2.0 * numpy.sqrt( 10.0 / 7.0 ) ),
                                +1.0 / 3.0 * numpy.sqrt( 5.0 + 2.0 * numpy.sqrt( 10.0 / 7.0 ) ) ] )
        w_gold = numpy.array( [ ( 322.0 - 13.0 * numpy.sqrt( 70.0 ) ) / 900.0,
                                ( 322.0 + 13.0 * numpy.sqrt( 70.0 ) ) / 900.0,
                                128.0 / 225.0,
                                ( 322.0 + 13.0 * numpy.sqrt( 70.0 ) ) / 900.0,
                                ( 322.0 - 13.0 * numpy.sqrt( 70.0 ) ) / 900.0, ] )
        [ qp, w ] = computeGaussLegendreQuadrature( 5 )
        self.assertTrue( numpy.allclose( qp, qp_gold ) )
        self.assertTrue( numpy.allclose( w, w_gold ) )
        
        
        
unittest.main()