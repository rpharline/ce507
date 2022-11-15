# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 15:24:50 2022

@author: rphar
"""

import numpy
import matplotlib
import bext
import Basis_Func as basis

def evaluateElementBernsteinBasisAtParamCoord( uspline, elem_id, param_coord ):
    elem_degree = bext.getElementDegree( uspline, elem_id ) # Get the degree of the element
    elem_bernstein_basis = numpy.zeros( elem_degree + 1 )
    for n in range( 0, elem_degree + 1 ):
        elem_bernstein_basis[n] = basis.evalBernsteinBasis1D(param_coord, elem_degree, n) # Evaluate the Bernstein basis at the parametric coordinate
    return elem_bernstein_basis

def evaluateElementSplineBasisAtParamCoord( uspline, elem_id, param_coord ):
    elem_ext_operator = bext.getElementExtractionOperator(uspline, elem_id) # Get the extraction operator of the element
    elem_bernstein_basis = evaluateElementBernsteinBasisAtParamCoord( uspline, elem_id, param_coord )
    elem_spline_basis = evaluateElementBernsteinBasisAtParamCoord(uspline, elem_id, param_coord) # Apply the extraction operator onto its Bernstein basis at the param coord
    return elem_spline_basis 

def plotUsplineBasis( uspline, color_by ):
    num_pts = 100
    xi = numpy.linspace( 0, 1, num_pts )
    num_elems = bext.getNumElems(uspline) # Get number of elements in the U-spline
    for elem_idx in range( 0, num_elems ):
        elem_id = bext.elemIdFromElemIdx(uspline, elem_idx) # Get the element id of the current element
        elem_domain = bext.getElementDomain(uspline, elem_id) # Get the domain of the current element
        elem_node_ids = bext.getElementNodeIds(uspline, elem_id) # Get the spline node ids of the current element
        elem_degree = bext.getElementDegree(uspline, elem_id)
        C_matrix = bext.getElementExtractionOperator(uspline, elem_id)
        x = numpy.linspace( elem_domain[0], elem_domain[1], num_pts )
        y = numpy.zeros( shape = ( elem_degree + 1, num_pts ) )
        for i in range( 0, num_pts ):
            y[:,i] = basis.evalUSplineBasis1D(x[i], C_matrix, elem_degree, elem_domain) # Evaluate the current element’s spline basis at the current coordinate
        # Do plotting for the current element
        for n in range( 0, elem_degree + 1 ):
            if color_by == "element":
                color = getLineColor( elem_idx )
            elif color_by == "node":
                color = getLineColor( elem_node_ids[n] )
            matplotlib.pyplot.plot( x, y[n,:], color =  color )
    matplotlib.pyplot.plot.show()

def getLineColor( idx ):
    colors = list( matplotlib.colors.TABLEAU_COLORS.keys() )
    num_colors = len( colors )
    mod_idx = idx % num_colors
    return matplotlib.colors.TABLEAU_COLORS[ colors[ mod_idx ] ]

uspline = bext.readBEXT( "quadratic_bspline.json" )
plotUsplineBasis( uspline, "element" )
plotUsplineBasis( uspline, "node" )