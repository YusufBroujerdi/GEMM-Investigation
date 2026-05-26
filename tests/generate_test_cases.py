import numpy as np
import argparse
from pathlib import Path

seed = 12134234522

generic_shapes = [
    [1,1,1],
    [1,5,1],
    [5,1,4],
    [1,7,9],
    [8,6,1],
    [8,1,8],
    [3,5,2],
    [17,9,13],
    [31,17,29],
    [33,35,31]
]


def create_test_case(m, k, n):

    rng = np.random.default_rng(seed)

    a = rng.normal(loc = 0, scale = 100, size=(m, k)).astype(np.float64)
    b = rng.normal(loc = 0, scale = 100, size=(k, n)).astype(np.float64)

    c = a @ b

    return {
        'left' : a,
        'right' : b,
        'output' : c
    }


def create_matrix_text(m, name):

    res = ''
    max_digit = max([len(f'{val:.10g}') for val in m.flat])
    rows = m.shape[0]
    cols = m.shape[1]

    res += f'mlk::Matrix<double> {name} {{{rows}, {cols},\n'
    res += f'    {{\n'

    for row in m:
        res += ' ' * 8
        for val in row:
            res += ' ' * (max_digit - len(f'{val:.10g}'))
            res += f'{val:.10g}, '
        res += '\n'

    res = res[:-2] + '\n'
    res += f'    }}\n'
    res += f'}};\n\n'

    return res


def create_test_case_text(shape, n):

    res = ''
    test_case = create_test_case(shape[0], shape[1], shape[2])

    left = f'left_{n}'
    right = f'right_{n}'
    output = f'output_{n}'
    test = f'test_{n}'

    res += create_matrix_text(test_case['left'], f'left_{n}')
    res += create_matrix_text(test_case['right'], f'right_{n}')
    res += create_matrix_text(test_case['output'], f'output_{n}')

    res += f'mlk::GemmTestCase<double> test_{n} {{left_{n}, right_{n}, output_{n}, "python_{n}"}};\n\n'

    return res


def create_python_tests_src(path):

    shape_no = len(generic_shapes)

    res = ''
    res += '#include <array>\n'
    res += '#include "mlkernels/matrix.hpp"\n\n'
    res += 'namespace gemm_python_tests {\n\n'

    for i in range(shape_no):
        res += create_test_case_text(generic_shapes[i], i + 1)

    res += f'std::array<mlk::GemmTestCase<double>, {shape_no}> tests {{'
    res += ''.join([f'test_{i+1}, ' for i in range(shape_no)])
    res = res[:-2]
    res += f'}};\n\n'
    res += f'}}'

    with open(path, 'w') as f:
        f.write(res)


def create_python_tests_header(path):

    shape_no = len(generic_shapes)

    res = ''
    res += f'#pragma once\n'
    res += f'#include <array>\n'
    res += f'#include "mlkernels/matrix.hpp"\n\n'
    res += f'namespace gemm_python_tests {{\n\n'
    res += f'    extern std::array<mlk::GemmTestCase<double>, {shape_no}> tests;\n'
    res += f'}}'

    with open(path, 'w') as f:
        f.write(res)


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("cpp_output", type = Path)
    parser.add_argument("hpp_output", type = Path)

    return parser.parse_args()


def main():

    args = parse_args()

    args.cpp_output.parent.mkdir(parents = True, exist_ok = True)
    args.hpp_output.parent.mkdir(parents = True, exist_ok = True)

    create_python_tests_header(args.hpp_output)
    create_python_tests_src(args.cpp_output)


if __name__ == '__main__':

    main()
