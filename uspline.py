# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 16:32:36 2022

@author: rphar
"""

import sys
import numpy

if __name__ == "CubitPythonInterpreter_2":
    # We are running within the Coreform Cubit application, cubit python module is already available
    pass
else:
    sys.path.append("your/path/to/cubit/bin")
    import cubit
    cubit.init([])

## MAIN FUNCTION
def make_uspline_mesh( spline_space, filename ):
  cubit.cmd( "reset" )
  build_geometry( spline_space )
  generate_bezier_mesh( spline_space )
  assign_uspline_params( spline_space )
  build_uspline()
  export_uspline( filename )

## SECONDARY FUNCTIONS
def build_geometry( spline_space ):
  cubit.cmd( f"create curve location {spline_space[’domain’][0]} 0 0 location {spline_space[’domain’][1]} 0 0" )

def generate_bezier_mesh( spline_space ):
  cubit.cmd( f"curve 1 interval {len(spline_space[’degree’])}" )
  cubit.cmd( "mesh curve 1" )

def assign_uspline_params( spline_space ):
  cubit.cmd( "set uspline curve 1 degree 1 continuity 0" )
  ien_array = get_ien_array()
  elem_id_list = tuple( ien_array.keys() )
  for eidx in range( 0, len( ien_array ) ):
    eid = elem_id_list[eidx]
    elem_degree = spline_space["degree"][eidx]
    cubit.cmd( f"set uspline edge {eid} degree {elem_degree}" )
    if eidx < ( len( ien_array ) - 1 ):
      interface_nid = ien_array[eid][1]
      interface_cont = spline_space["continuity"][eidx + 1]
      cubit.cmd( f"set uspline node {interface_nid} continuity {interface_cont}" )

def build_uspline():
  cubit.cmd( "build uspline curve 1 as 1" )
  cubit.cmd( "fit uspline 1" )

def export_uspline( filename ):
  cubit.cmd( f"export uspline 1 json ’{filename}’" )

## UTILITY FUNCTIONS
def get_num_elems():
  ien_array = get_ien_array()
  return max( ien_array.keys() )

def get_ien_array():
  elem_list = cubit.parse_cubit_list( "edge", "in curve 1" )
  ien_array = {}
  for eid in elem_list:
    ien_array[eid] = cubit.get_connectivity( "edge", eid )
  ien_array = sort_element_nodes( ien_array )
  return ien_array

def get_elem_centers( ien_array ):
  xc = []
  for eid in ien_array:
    x0 = cubit.get_nodal_coordinates( ien_array[eid][0] )[0]
    x1 = cubit.get_nodal_coordinates( ien_array[eid][1] )[0]
    xc.append( ( x0 + x1 ) / 2.0 )
  xc = numpy.array( xc )
  return xc

def get_ordered_elem_list():
  ien_array = get_ien_array()
  sort_idx = numpy.argsort( get_elem_centers( ien_array ) )
  sorted_elems = list( ien_array.keys() )
  return sorted_elems

def sort_element_nodes( ien_array ):
  for eid in ien_array:
    x0 = cubit.get_nodal_coordinates( ien_array[eid][0] )[0]
    x1 = cubit.get_nodal_coordinates( ien_array[eid][1] )[0]
    if x1 < x0:
      ien_array[eid] = tuple( reversed( ien_array[eid] ) )
  return ien_array