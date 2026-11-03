#!/usr/bin/env python3
"""Create sample MATLAB files for testing."""

import numpy as np
import scipy.io


def create_sample_mat_v5(output_path):
    """Create a sample MAT-File v5 with various data types."""

    # Create sample data
    data = {
        # Scalar
        'scalar_value': 42,

        # Vectors
        'row_vector': np.array([[1, 2, 3, 4, 5]]),
        'col_vector': np.array([[1], [2], [3], [4], [5]]),

        # Matrices
        'matrix_2d': np.random.randn(3, 4),
        'matrix_3d': np.random.randn(2, 3, 4),

        # Complex numbers
        'complex_data': np.array([1+2j, 3+4j, 5+6j]),

        # Strings
        'text_string': 'Hello from MATLAB!',

        # Logical/Boolean
        'logical_array': np.array([True, False, True, False], dtype=bool),

        # Integer types
        'int_array': np.array([1, 2, 3, 4, 5], dtype=np.int32),
        'uint_array': np.array([10, 20, 30, 40, 50], dtype=np.uint16),

        # Floating point
        'float_data': np.array([1.1, 2.2, 3.3, 4.4, 5.5], dtype=np.float32),
        'double_data': np.array([1.111, 2.222, 3.333], dtype=np.float64),

        # Sparse matrix (commented out if scipy.sparse not needed for basic test)
        # 'sparse_matrix': scipy.sparse.csr_matrix([[1, 0, 0], [0, 2, 0], [0, 0, 3]]),
    }

    # Save to MAT file
    scipy.io.savemat(output_path, data, format='5')
    print(f"Created MAT-File v5: {output_path}")
    print(f"Variables: {list(data.keys())}")


def create_sample_mat_v7(output_path):
    """Create a sample MAT-File v7 (compressed)."""

    data = {
        'sensor_data': np.random.randn(100, 3),
        'timestamps': np.linspace(0, 10, 100),
        'metadata': {
            'device_id': 'SENSOR_001',
            'location': 'Lab A',
            'sample_rate': 100.0,
        },
    }

    # Note: scipy doesn't directly support saving nested structs easily
    # We'll save a flattened version
    flat_data = {
        'sensor_data': data['sensor_data'],
        'timestamps': data['timestamps'],
        'device_id': 'SENSOR_001',
        'location': 'Lab A',
        'sample_rate': 100.0,
    }

    scipy.io.savemat(output_path, flat_data, format='5', do_compression=True)
    print(f"\nCreated MAT-File v7 (compressed): {output_path}")
    print(f"Variables: {list(flat_data.keys())}")


if __name__ == "__main__":
    import sys

    # Create v5 format
    v5_output = sys.argv[1] if len(sys.argv) > 1 else "samples/sample_v5.mat"
    create_sample_mat_v5(v5_output)

    # Create v7 format (compressed)
    v7_output = sys.argv[2] if len(sys.argv) > 2 else "samples/sample_v7.mat"
    create_sample_mat_v7(v7_output)
