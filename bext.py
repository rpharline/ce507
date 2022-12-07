# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 15:28:47 2022

@author: rphar
"""

import json
import numpy

## IMPORT USER-CREATED MODULES
import Basis_Func as basis

def readBEXT( filename ):
    f = open( filename, "r" )
    uspline = json.load( f )
    f.close()
    return uspline

def getNumElems( uspline ):
    return uspline["num_elems"]

def getNumVertices( uspline ):
    return uspline["num_vertices"]

def getDomain( uspline ):
    nodes = getSplineNodes( uspline )
    return [ min( nodes[:,0] ), max( nodes[:,0] ) ]

def getNodeIdNearPoint( uspline, point ):
    spline_nodes = getSplineNodes( uspline )[:,0]
    node_dist = numpy.sqrt( ( spline_nodes - point )**2.0 )
    return numpy.argmin( node_dist )

def getNumNodes( uspline ):
    return getSplineNodes( uspline ).shape[0]

def elemIdFromElemIdx( uspline, elem_idx ):
    element_blocks = uspline["elements"]["element_blocks"]
    elem_id = element_blocks[ elem_idx ]["us_cid"]
    return elem_id

def elemIdxFromElemId( uspline, elem_id ):
    element_blocks = uspline["elements"]["element_blocks"]
    for elem_idx in range( 0, len( element_blocks ) ):
        if element_blocks[ elem_idx ]["us_cid"] == elem_id:
            return elem_idx

def getElementDegree( uspline, elem_id ):
    elem_idx = elemIdxFromElemId( uspline, elem_id )
    return len( uspline["elements"]["element_blocks"][elem_idx]["node_ids"] ) - 1

def getElementDomain( uspline, elem_id ):
    elem_bezier_nodes = getElementBezierNodes( uspline, elem_id )
    elem_domain = [ min( elem_bezier_nodes[:,0] ), max( elem_bezier_nodes[:,0] ) ]
    return elem_domain

def getElementNodeIds( uspline, elem_id ):
    elem_idx = elemIdxFromElemId( uspline, elem_id )
    elem_node_ids = numpy.array( uspline["elements"]["element_blocks"][elem_idx]["node_ids"] )
    return elem_node_ids

def getElementNodes( uspline, elem_id ):
    elem_node_ids = getElementNodeIds( uspline, elem_id )
    spline_nodes = getSplineNodes( uspline )
    elem_nodes = spline_nodes[elem_node_ids, 0:-1]
    return elem_nodes

def getSplineNodes( uspline ):
    return numpy.array( uspline["nodes"] )

def getCoefficientVectors( uspline ):
    coeff_vectors_list = uspline["coefficients"]["dense_coefficient_vectors"]
    coeff_vectors = {}
    for i in range( 0, len( coeff_vectors_list ) ):
        coeff_vectors[i] = coeff_vectors_list[i]["components"]
    return coeff_vectors

def getElementCoefficientVectorIds( uspline, elem_id ):
    elem_idx = elemIdxFromElemId( uspline, elem_id )
    return uspline["elements"]["element_blocks"][elem_idx]["coeff_vector_ids"]

def getVertexConnectivity( uspline ):
    return uspline["vertex_connectivity"]

def getElementExtractionOperator( uspline, elem_id ):
    coeff_vectors = getCoefficientVectors( uspline )    
    coeff_vector_ids = getElementCoefficientVectorIds( uspline, elem_id )
    C = numpy.zeros( shape = (len( coeff_vector_ids ), len( coeff_vector_ids ) ), dtype = "double" )
    for n in range( 0, len( coeff_vector_ids ) ):
        C[n,:] = coeff_vectors[ coeff_vector_ids[n] ]
    return C

def getElementBezierNodes( uspline, elem_id ):
    elem_nodes = getElementNodes( uspline, elem_id )
    C = getElementExtractionOperator( uspline, elem_id )
    element_bezier_node_coords = C.T @ elem_nodes
    return element_bezier_node_coords

def getElementBezierVertices( uspline, elem_id ):
    element_bezier_node_coords = getElementBezierNodes( uspline, elem_id )
    vertex_connectivity = getVertexConnectivity( uspline )
    vertex_coords = numpy.array( [ element_bezier_node_coords[0], element_bezier_node_coords[-1] ] )
    return vertex_coords

def getBezierNodes( uspline ):
    bezier_nodes = []
    for elem_idx in range( 0, getNumElems( uspline ) ):
        elem_id = elemIdFromElemIdx( uspline, elem_idx )
        elem_bezier_nodes = getElementBezierNodes( uspline, elem_id )
        bezier_nodes.append( elem_bezier_nodes )
    bezier_nodes = uniquetol( bezier_nodes, 1e-12 )
    return bezier_nodes

def getElementIdContainingPoint( uspline, point ):
    num_elems = getNumElems( uspline )
    for elem_idx in range( 0, num_elems ):
        elem_id = elemIdFromElemIdx( uspline, elem_idx )
        elem_domain = getElementDomain( uspline, elem_id )
        if ( ( point >= elem_domain[0] ) and ( point <= elem_domain[1] ) ):
            return elem_id
    raise Exception( "ELEMENT_CONTAINING_POINT_NOT_FOUND" )

def uniquetol( input_array, tol ):
    equalityArray = numpy.zeros( len( input_array ), dtype="bool" )
    for i in range( 0, len( input_array) ):
        for j in range( i+1, len( input_array ) ):
            if abs( input_array[ i ] - input_array[ j ] ) <= tol:
                equalityArray[i] = True
    return input_array[ ~equalityArray ]