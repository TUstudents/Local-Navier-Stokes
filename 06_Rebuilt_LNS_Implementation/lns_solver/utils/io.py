"""
Input/output utilities for LNS solver.

This module provides classes for reading and writing LNS simulation data
in various formats (HDF5, NetCDF, VTK).
"""

from typing import Dict, List, Optional, Union, Any, Tuple
import numpy as np
import h5py
from pathlib import Path
import logging

from lns_solver.core.grid import LNSGrid
from lns_solver.core.state import LNSState

logger = logging.getLogger(__name__)


class LNSDataWriter:
    """
    Data writer for LNS simulation results.
    
    Supports multiple output formats:
    - HDF5: Efficient binary format for large datasets
    - VTK: For visualization in ParaView/VisIt
    - CSV: Simple text format for small datasets
    """
    
    def __init__(self, output_format: str = 'hdf5', compression: bool = True):
        """
        Initialize data writer.
        
        Args:
            output_format: Output format ('hdf5', 'vtk', 'csv')
            compression: Whether to use compression
        """
        self.output_format = output_format.lower()
        self.compression = compression
        
        if self.output_format not in ['hdf5', 'vtk', 'csv']:
            raise ValueError(f"Unsupported format: {output_format}")
    
    def write_state(
        self,
        filename: Union[str, Path],
        state: LNSState,
        time: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Write state data to file.
        
        Args:
            filename: Output filename
            state: LNS state to write
            time: Simulation time
            metadata: Additional metadata
        """
        filename = Path(filename)
        
        if self.output_format == 'hdf5':
            self._write_hdf5(filename, state, time, metadata)
        elif self.output_format == 'vtk':
            self._write_vtk(filename, state, time, metadata)
        elif self.output_format == 'csv':
            self._write_csv(filename, state, time, metadata)
        
        logger.info(f"Wrote state data to {filename}")
    
    def _write_hdf5(
        self,
        filename: Path,
        state: LNSState,
        time: float,
        metadata: Optional[Dict[str, Any]]
    ) -> None:
        """Write data in HDF5 format."""
        with h5py.File(filename, 'w') as f:
            # Grid information
            grid_group = f.create_group('grid')
            grid_group.attrs['ndim'] = state.grid.ndim
            grid_group.attrs['nx'] = state.grid.nx
            if state.grid.ndim >= 2:
                grid_group.attrs['ny'] = state.grid.ny
            if state.grid.ndim >= 3:
                grid_group.attrs['nz'] = state.grid.nz
            
            # Coordinates
            if self.compression:
                grid_group.create_dataset('x', data=state.grid.x, compression='gzip')
                if state.grid.ndim >= 2:
                    grid_group.create_dataset('y', data=state.grid.y, compression='gzip')
                if state.grid.ndim >= 3:
                    grid_group.create_dataset('z', data=state.grid.z, compression='gzip')
            else:
                grid_group.create_dataset('x', data=state.grid.x)
                if state.grid.ndim >= 2:
                    grid_group.create_dataset('y', data=state.grid.y)
                if state.grid.ndim >= 3:
                    grid_group.create_dataset('z', data=state.grid.z)
            
            # State data
            state_group = f.create_group('state')
            state_group.attrs['time'] = time
            state_group.attrs['n_variables'] = state.n_variables
            
            # Conservative variables
            if self.compression:
                state_group.create_dataset('Q', data=state.Q, compression='gzip')
            else:
                state_group.create_dataset('Q', data=state.Q)
            
            # Primitive variables
            try:
                primitives = state.get_primitive_variables()
                prim_group = state_group.create_group('primitives')
                
                for var_name, var_data in primitives.items():
                    if self.compression:
                        prim_group.create_dataset(var_name, data=var_data, compression='gzip')
                    else:
                        prim_group.create_dataset(var_name, data=var_data)
            except Exception as e:
                logger.warning(f"Could not write primitive variables: {e}")
            
            # Metadata
            if metadata:
                meta_group = f.create_group('metadata')
                for key, value in metadata.items():
                    meta_group.attrs[key] = value
    
    def _write_vtk(
        self,
        filename: Path,
        state: LNSState,
        time: float,
        metadata: Optional[Dict[str, Any]]
    ) -> None:
        """Write data in VTK format (simplified)."""
        # This is a basic VTK writer - full implementation would use PyVista or VTK library
        logger.warning("VTK writer not fully implemented - using basic format")
        
        with open(filename.with_suffix('.vtk'), 'w') as f:
            f.write("# vtk DataFile Version 3.0\n")
            f.write(f"LNS simulation data at t={time}\n")
            f.write("ASCII\n")
            f.write("DATASET STRUCTURED_GRID\n")
            
            if state.grid.ndim == 1:
                f.write(f"DIMENSIONS {state.grid.nx} 1 1\n")
                f.write(f"POINTS {state.grid.nx} float\n")
                for x in state.grid.x:
                    f.write(f"{x} 0.0 0.0\n")
            elif state.grid.ndim == 2:
                f.write(f"DIMENSIONS {state.grid.nx} {state.grid.ny} 1\n")
                f.write(f"POINTS {state.grid.nx * state.grid.ny} float\n")
                for j in range(state.grid.ny):
                    for i in range(state.grid.nx):
                        f.write(f"{state.grid.x[i]} {state.grid.y[j]} 0.0\n")
            
            # Point data
            f.write(f"POINT_DATA {len(state.Q)}\n")
            
            # Write primitive variables
            try:
                primitives = state.get_primitive_variables()
                for var_name, var_data in primitives.items():
                    f.write(f"SCALARS {var_name} float 1\n")
                    f.write("LOOKUP_TABLE default\n")
                    for value in var_data:
                        f.write(f"{value}\n")
            except Exception as e:
                logger.warning(f"Could not write VTK primitive variables: {e}")
    
    def _write_csv(
        self,
        filename: Path,
        state: LNSState,
        time: float,
        metadata: Optional[Dict[str, Any]]
    ) -> None:
        """Write data in CSV format."""
        with open(filename.with_suffix('.csv'), 'w') as f:
            # Header
            f.write(f"# LNS simulation data at t={time}\n")
            
            if state.grid.ndim == 1:
                header = "x,density,velocity,pressure,temperature,heat_flux_x,stress_xx\n"
            elif state.grid.ndim == 2:
                header = "x,y,density,velocity_x,velocity_y,pressure,temperature,heat_flux_x,heat_flux_y,stress_xx,stress_yy,stress_xy\n"
            else:
                header = "# 3D CSV output not implemented\n"
            
            f.write(header)
            
            # Data
            try:
                primitives = state.get_primitive_variables()
                
                if state.grid.ndim == 1:
                    for i, x in enumerate(state.grid.x):
                        line = f"{x},{primitives['density'][i]},{primitives['velocity'][i]},"
                        line += f"{primitives['pressure'][i]},{primitives['temperature'][i]},"
                        line += f"{primitives['heat_flux_x'][i]},{primitives['stress_xx'][i]}\n"
                        f.write(line)
                
                elif state.grid.ndim == 2:
                    idx = 0
                    for j in range(state.grid.ny):
                        for i in range(state.grid.nx):
                            x, y = state.grid.x[i], state.grid.y[j]
                            line = f"{x},{y},{primitives['density'][idx]},"
                            line += f"{primitives['velocity_x'][idx]},{primitives['velocity_y'][idx]},"
                            line += f"{primitives['pressure'][idx]},{primitives['temperature'][idx]},"
                            line += f"{primitives['heat_flux_x'][idx]},{primitives['heat_flux_y'][idx]},"
                            line += f"{primitives['stress_xx'][idx]},{primitives['stress_yy'][idx]},{primitives['stress_xy'][idx]}\n"
                            f.write(line)
                            idx += 1
                            
            except Exception as e:
                logger.error(f"Could not write CSV data: {e}")


class LNSDataReader:
    """
    Data reader for LNS simulation results.
    
    Supports reading from multiple formats to reconstruct LNSState objects.
    """
    
    def __init__(self):
        """Initialize data reader."""
        pass
    
    def read_state(
        self,
        filename: Union[str, Path],
        grid: Optional[LNSGrid] = None
    ) -> Tuple[LNSState, float, Dict[str, Any]]:
        """
        Read state data from file.
        
        Args:
            filename: Input filename
            grid: Optional grid (will be read from file if not provided)
            
        Returns:
            Tuple of (state, time, metadata)
        """
        filename = Path(filename)
        
        if filename.suffix == '.h5' or filename.suffix == '.hdf5':
            return self._read_hdf5(filename, grid)
        elif filename.suffix == '.csv':
            return self._read_csv(filename, grid)
        else:
            raise ValueError(f"Unsupported file format: {filename.suffix}")
    
    def _read_hdf5(
        self,
        filename: Path,
        grid: Optional[LNSGrid]
    ) -> Tuple[LNSState, float, Dict[str, Any]]:
        """Read data from HDF5 format."""
        with h5py.File(filename, 'r') as f:
            # Read grid if not provided
            if grid is None:
                grid_group = f['grid']
                ndim = grid_group.attrs['ndim']
                nx = grid_group.attrs['nx']
                
                coordinates = {'x': grid_group['x'][:]}
                spacing = {'dx': coordinates['x'][1] - coordinates['x'][0] if nx > 1 else 1.0}
                
                if ndim >= 2:
                    ny = grid_group.attrs['ny']
                    coordinates['y'] = grid_group['y'][:]
                    spacing['dy'] = coordinates['y'][1] - coordinates['y'][0] if ny > 1 else 1.0
                
                if ndim >= 3:
                    nz = grid_group.attrs['nz']
                    coordinates['z'] = grid_group['z'][:]
                    spacing['dz'] = coordinates['z'][1] - coordinates['z'][0] if nz > 1 else 1.0
                
                grid = LNSGrid(ndim, coordinates, spacing)
            
            # Read state
            state_group = f['state']
            time = state_group.attrs['time']
            n_variables = state_group.attrs['n_variables']
            
            Q_data = state_group['Q'][:]
            
            # Create state object
            state = LNSState(grid, n_variables)
            state.Q = Q_data
            
            # Read metadata
            metadata = {}
            if 'metadata' in f:
                meta_group = f['metadata']
                for key in meta_group.attrs:
                    metadata[key] = meta_group.attrs[key]
        
        logger.info(f"Read state data from {filename}")
        return state, time, metadata
    
    def _read_csv(
        self,
        filename: Path,
        grid: Optional[LNSGrid]
    ) -> Tuple[LNSState, float, Dict[str, Any]]:
        """Read data from CSV format (basic implementation)."""
        # This is a simplified CSV reader
        logger.warning("CSV reader is basic - may not handle all cases")
        
        data = np.loadtxt(filename, delimiter=',', skiprows=2)
        
        if grid is None:
            # Try to reconstruct grid from data
            if data.shape[1] >= 7:  # 1D case
                x = data[:, 0]
                dx = x[1] - x[0] if len(x) > 1 else 1.0
                grid = LNSGrid(1, {'x': x}, {'dx': dx})
            else:
                raise ValueError("Cannot reconstruct grid from CSV - please provide grid")
        
        # Create state (this is simplified and may not work for all cases)
        n_variables = 5 if grid.ndim == 1 else (9 if grid.ndim == 2 else 13)
        state = LNSState(grid, n_variables)
        
        # This would need proper implementation based on CSV format
        logger.warning("CSV state reconstruction not fully implemented")
        
        return state, 0.0, {}