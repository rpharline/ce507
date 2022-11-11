# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 15:03:05 2022

@author: rphar
"""

import unittest
import math
import numpy
import sympy
import sys

from matplotlib import pyplot as plt


def taylorExpansion( fun, a, order ):
    x = sympy.Symbol('x')
    i = 0
    p = 0
    
    while i <= order:
        p = p + (fun.diff(x,i).subs(x,a))/(sympy.factorial(i))*(x-a)**i
        i += 1
    return p

def evaluateMonomialBasis1D(degree, variate):
    
    value = 0
    for i in range(0,degree+1):
        value = variate**i    
    
    return value

def evalLegendreBasis1D(degree,variate):
    if degree == 0:
        value = 1
    elif degree == 1:
        value = variate
    else:
        i = degree - 1
        term_1 = -i * evalLegendreBasis1D(i-1, variate)
        term_2 = (2*i + 1) * variate * evalLegendreBasis1D(i, variate)
        value = (term_2 + term_1) / (i+1)
    return value


def evaluateLagrangeBasis1D(variate,degree,basis_idx):
    
    nodes = numpy.linspace(-1,1,degree+1)
    
    control_node = nodes[basis_idx]
    
    value = 1
    for j in range(0,degree+1):
        
        if j == basis_idx:
            continue
        
        else:
            value *= (variate - nodes[j]) / (control_node - nodes[j])
        
    return value

def evalBernsteinBasis1D(variate, degree, basis_idx):
    #change of basis from [-1,1] to [0,1]
    variate = (1 + variate)*0.5
    
    coeff = math.comb(degree, basis_idx)
    
    value = coeff * (variate**(basis_idx)) * (1-variate)**(degree - basis_idx)
    
    return value

# graph out functions
# x = numpy.linspace(-1,1,100)
# for i in range(0,len(x)):
#     y = evaluateBernsteinBasis1D(x,2,1)

# plt.plot(x,y)

#tests monomial basis
# class Test_evaluateMonomialBasis1D( unittest.TestCase ):
#    def test_basisAtBounds( self ):
#        self.assertAlmostEqual( first = evaluateMonomialBasis1D( degree = 0, variate = 0 ), second = 1.0, delta = 1e-12 )
#        for p in range( 1, 11 ):
#            self.assertAlmostEqual( first = evaluateMonomialBasis1D( degree = p, variate = 0 ), second = 0.0, delta = 1e-12 )
#            self.assertAlmostEqual( first = evaluateMonomialBasis1D( degree = p, variate = 1 ), second = 1.0, delta = 1e-12 )

#    def test_basisAtMidpoint( self ):
#        for p in range( 0, 11 ):
#            self.assertAlmostEqual( first = evaluateMonomialBasis1D( degree = p, variate = 0.5 ), second = 1 / ( 2**p ), delta = 1e-12 )

# #tests Legendre Basis
# class Test_evalLegendreBasis1D( unittest.TestCase ):
#     def test_basisAtBounds( self ):
#         for p in range( 0, 2 ):
#             if ( p % 2 == 0 ):
#                 self.assertAlmostEqual( first = evalLegendreBasis1D( degree = p, variate = -1 ), second = +1.0, delta = 1e-12 )
#             else:
#                 self.assertAlmostEqual( first = evalLegendreBasis1D( degree = p, variate = -1 ), second = -1.0, delta = 1e-12 )
#             self.assertAlmostEqual( first = evalLegendreBasis1D( degree = p, variate = +1 ), second = 1.0, delta = 1e-12 )

#     def test_constant( self ):
#         for x in numpy.linspace( -1, 1, 100 ):
#             self.assertAlmostEqual( first = evalLegendreBasis1D( degree = 0, variate = x ), second = 1.0, delta = 1e-12 )

#     def test_linear( self ):
#         for x in numpy.linspace( -1, 1, 100 ):
#             self.assertAlmostEqual( first = evalLegendreBasis1D( degree = 1, variate = x ), second = x, delta = 1e-12 )

#     def test_quadratic_at_roots( self ):
#         self.assertAlmostEqual( first = evalLegendreBasis1D( degree = 2, variate = -1.0 / math.sqrt(3.0) ), second = 0.0, delta = 1e-12 )
#         self.assertAlmostEqual( first = evalLegendreBasis1D( degree = 2, variate = +1.0 / math.sqrt(3.0) ), second = 0.0, delta = 1e-12 )

#     def test_cubic_at_roots( self ):
#         self.assertAlmostEqual( first = evalLegendreBasis1D( degree = 3, variate = -math.sqrt( 3 / 5 ) ), second = 0.0, delta = 1e-12 )
#         self.assertAlmostEqual( first = evalLegendreBasis1D( degree = 3, variate = 0 ), second = 0.0, delta = 1e-12 )
#         self.assertAlmostEqual( first = evalLegendreBasis1D( degree = 3, variate = +math.sqrt( 3 / 5 ) ), second = 0.0, delta = 1e-12 )
   
# #tests lagrange basis
# class Test_evaluateLagrangeBasis1D( unittest.TestCase ):
#     def test_linearLagrange( self ):
#         self.assertAlmostEqual( first = evaluateLagrangeBasis1D( variate = -1, degree = 1, basis_idx = 0 ), second = 1.0, delta = 1e-12 )
#         self.assertAlmostEqual( first = evaluateLagrangeBasis1D( variate = -1, degree = 1, basis_idx = 1 ), second = 0.0, delta = 1e-12 )
#         self.assertAlmostEqual( first = evaluateLagrangeBasis1D( variate = +1, degree = 1, basis_idx = 0 ), second = 0.0, delta = 1e-12 )
#         self.assertAlmostEqual( first = evaluateLagrangeBasis1D( variate = +1, degree = 1, basis_idx = 1 ), second = 1.0, delta = 1e-12 )

#     def test_quadraticLagrange( self ):
#         self.assertAlmostEqual( first = evaluateLagrangeBasis1D( variate = -1, degree = 2, basis_idx = 0 ), second = 1.0, delta = 1e-12 )
#         self.assertAlmostEqual( first = evaluateLagrangeBasis1D( variate = -1, degree = 2, basis_idx = 1 ), second = 0.0, delta = 1e-12 )
#         self.assertAlmostEqual( first = evaluateLagrangeBasis1D( variate = -1, degree = 2, basis_idx = 2 ), second = 0.0, delta = 1e-12 )
#         self.assertAlmostEqual( first = evaluateLagrangeBasis1D( variate =  0, degree = 2, basis_idx = 0 ), second = 0.0, delta = 1e-12 )
#         self.assertAlmostEqual( first = evaluateLagrangeBasis1D( variate =  0, degree = 2, basis_idx = 1 ), second = 1.0, delta = 1e-12 )
#         self.assertAlmostEqual( first = evaluateLagrangeBasis1D( variate =  0, degree = 2, basis_idx = 2 ), second = 0.0, delta = 1e-12 )
#         self.assertAlmostEqual( first = evaluateLagrangeBasis1D( variate = +1, degree = 2, basis_idx = 0 ), second = 0.0, delta = 1e-12 )
#         self.assertAlmostEqual( first = evaluateLagrangeBasis1D( variate = +1, degree = 2, basis_idx = 1 ), second = 0.0, delta = 1e-12 )
#         self.assertAlmostEqual( first = evaluateLagrangeBasis1D( variate = +1, degree = 2, basis_idx = 2 ), second = 1.0, delta = 1e-12 )
 
# #tests Bernstein basis
# class Test_evaluateBernsteinBasis1D( unittest.TestCase ):
#     def test_linearBernstein( self ):
#         self.assertAlmostEqual( first = evaluateBernsteinBasis1D( variate = -1, degree = 1, basis_idx = 0 ), second = 1.0, delta = 1e-12 )
#         self.assertAlmostEqual( first = evaluateBernsteinBasis1D( variate = -1, degree = 1, basis_idx = 1 ), second = 0.0, delta = 1e-12 )
#         self.assertAlmostEqual( first = evaluateBernsteinBasis1D( variate = +1, degree = 1, basis_idx = 0 ), second = 0.0, delta = 1e-12 )
#         self.assertAlmostEqual( first = evaluateBernsteinBasis1D( variate = +1, degree = 1, basis_idx = 1 ), second = 1.0, delta = 1e-12 )

#     def test_quadraticBernstein( self ):
#         self.assertAlmostEqual( first = evaluateBernsteinBasis1D( variate = -1, degree = 2, basis_idx = 0 ), second = 1.00, delta = 1e-12 )
#         self.assertAlmostEqual( first = evaluateBernsteinBasis1D( variate = -1, degree = 2, basis_idx = 1 ), second = 0.00, delta = 1e-12 )
#         self.assertAlmostEqual( first = evaluateBernsteinBasis1D( variate = -1, degree = 2, basis_idx = 2 ), second = 0.00, delta = 1e-12 )
#         self.assertAlmostEqual( first = evaluateBernsteinBasis1D( variate =  0, degree = 2, basis_idx = 0 ), second = 0.25, delta = 1e-12 )
#         self.assertAlmostEqual( first = evaluateBernsteinBasis1D( variate =  0, degree = 2, basis_idx = 1 ), second = 0.50, delta = 1e-12 )
#         self.assertAlmostEqual( first = evaluateBernsteinBasis1D( variate =  0, degree = 2, basis_idx = 2 ), second = 0.25, delta = 1e-12 )
#         self.assertAlmostEqual( first = evaluateBernsteinBasis1D( variate = +1, degree = 2, basis_idx = 0 ), second = 0.00, delta = 1e-12 )
#         self.assertAlmostEqual( first = evaluateBernsteinBasis1D( variate = +1, degree = 2, basis_idx = 1 ), second = 0.00, delta = 1e-12 )
#         self.assertAlmostEqual( first = evaluateBernsteinBasis1D( variate = +1, degree = 2, basis_idx = 2 ), second = 1.00, delta = 1e-12 )
        
# unittest.main()