from enum import Enum
from pathlib import Path
import subprocess
import pandas as pd

class FloatType(Enum):
    float_t = 'float'
    double_t = 'double'

class Kernel(Enum):
    naive = 'gemm_naive'
    reordered = 'gemm_reordered'
    tiled_a = 'gemm_tiled_a'
    tiled_b = 'gemm_tiled_b'
    multithreaded_a = 'gemm_multithreaded_a'
    multithreaded_b = 'gemm_multithreaded_b'

class Spec:

    def __init__(
        self,
        *,
        case_name = "generic",
        m = 2048,
        k = 2048,
        n = 2048,
        float_type = FloatType.float_t,
        kernel = Kernel.reordered,
        block_size = 256,
        threads = 12,
        repetitions = 10,
        seed = 12345
    ):

        if type(float_type) is not FloatType or type(kernel) is not Kernel:
            raise TypeError('Use appropriate enums for spec')

        if any([type(col) is not int for col in [m,n,k,block_size,threads,repetitions,seed]]):
            raise TypeError('Expected int but didn\'t get one')

        if type(case_name) is not str:
            raise TypeError('case name must be str')

        self.case_name = case_name
        self.m = m
        self.k = k
        self.n = n
        self.float_type = float_type
        self.kernel = kernel
        self.block_size = block_size
        self.threads = threads
        self.repetitions = repetitions
        self.seed = seed


    def __str__(self):

        return ','.join([str(i) for i in vars(self).values()])


class Specs(list):

    def __init__(self, items = ()):

        for item in items:
            if type(item) is not Spec:
                raise TypeError('class must consist of Spec objects')

        super().__init__(items)


    def __str__(self):

        header = 'case_name,m,n,k,float_type,kernel,block_size,threads,repetitions,seed\n'
        rows = '\n'.join([str(i) for i in self])
        return header + rows


def evaluate_specs(specs):

    if type(specs) is not Specs:
        raise TypeError('argument must have type Specs')

    spec_csv_path = Path(__file__).resolve().parent / 'spec.csv'
    results_csv_path = Path(__file__).resolve().parent / 'results.csv'
    bench_path = Path(__file__).resolve().parent / 'bench_gemm'
    with open(spec_csv_path, 'w') as spec_csv:
        spec_csv.write(str(specs))

    subprocess.run([bench_path, spec_csv_path, results_csv_path], check = True)

    df = pd.read_csv(results_csv_path, dtype = {
        'case_name' : 'string',
        'M' : 'int64',
        'K' : 'int64',
        'N' : 'int64',
        'float_type' : 'string',
        'kernel' : 'string',
        'block_size' : 'int64',
        'threads' : 'int64',
        'repetitions' : 'int64',
        'seed' : 'int64',
        'time_ms_min' : 'float64',
        'time_ms_max' : 'float64',
        'time_ms_mean' : 'float64',
        'gflops_per_second' : 'float64',
        'max_abs_error' : 'float64',
        'max_rel_error' : 'float64',
        'mean_abs_error' : 'float64',
        'validation_result' : 'bool'
    }
    )

    df['float_type'] = df['float_type'].map(FloatType)
    df['kernel'] = df['kernel'].map(Kernel)

    return df


if __name__ == '__main__':

    evaluate_specs(Specs(
        [
            Spec(
                kernel = kernel_type,
                case_name = kernel_type.name,
                m = 2048,
                n = 2048,
                k = 2048
            ) for kernel_type in Kernel
        ]
    ))
