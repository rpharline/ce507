# -*- coding: utf-8 -*-
"""
Created on Sat Sep 24 14:01:49 2022

@author: rphar
"""

import Basis_Func as BF

import numpy as np
from matplotlib import pyplot as plt
import sympy
from scipy import integrate

#Question 32 Part 1
# x = sympy.Symbol('x')
# fun = sympy.exp(x)
# Tx = BF.taylorExpansion(fun, 0, 10)
# print(Tx)

# sympy.plot(BF.taylorExpansion(fun,0,0),BF.taylorExpansion(fun, 0, 1),\
#            fun,(x,-1,1))
# sympy.plot(taylorExpansion(fun, a, 0),taylorExpansion(fun, a, 1) \
#            ,taylorExpansion(fun, a, 3),taylorExpansion(fun, a, 5),\
#            taylorExpansion(fun, a, 7),taylorExpansion(fun, a, 9),\
#            taylorExpansion(fun, a, 11),fx,(x,-2,2))

# print(sympy.integrate(Tx,(x,-1,0)))

#Question 32 Part 2
# x = sympy.Symbol('x')
# order = sympy.Symbol('order')

# fun = sympy.sin(x*sympy.pi)
# taylor = BF.taylorExpansion(fun, 0, 10)
# abs_diff = (sympy.Abs(fun-taylor))


# abs_int_diff = sympy.Abs(fun-taylor)

# x = sympy.Symbol('x')
# fun = sympy.erfc(x)

# error = []
# degree = list( range( 0, 11 ) )
# for p in degree:
#     t = BF.taylorExpansion( fun, 0, p )
#     error.append(integrate.quad( sympy.lambdify( x, abs( t - fun ) ), -2, 2, limit = 1000 )[0])
                 
# plt.yscale("log")
# plt.plot(error)




# integral = sympy.integrate(errorfunc,(x-1,1))
# sympy.plot(integral,(x,0,1),yscale = 'log')


#Question 33
#plot monomial basis up to 10 degrees
# x = sympy.Symbol('x')
# degree = list(range(0,11))
# functions = []
# for i in degree:
#     functions.append(BF.evaluateMonomialBasis1D(i, x))

# sympy.plot(functions[0],functions[1],functions[2],functions[3],functions[4]\
#            ,functions[5],functions[6],functions[7],functions[8],functions[9],\
#                functions[10],(x,0,1))



#Question 43
# x = sympy.Symbol('x')
# basis_1 = 0
# basis_2 = 4/3 * BF.evaluateBernsteinBasis1D(x, 3, 1)
# basis_3 = 0
# basis_4 = 4 * BF.evaluateBernsteinBasis1D(x, 3, 3)
# total = basis_1 + basis_2 + basis_3 + basis_4
# sympy.plot(basis_1,basis_2,basis_3,basis_4,total,(x,-1,1))