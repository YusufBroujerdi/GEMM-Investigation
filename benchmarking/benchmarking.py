from enum import Enum
from pathlib import Path
import subprocess
import pandas as pd

class FloatType(Enum):
    float_t = 0
    double_t = 1

class Kernel(Enum):
    naive = 0
    reordered = 1

class Spec:

    def __init__(
        self,
        *,
        case_name = "generic",
        m = 1024,
        k = 1024,
        n = 1024,
        float_type = FloatType.float_t,
        kernel = Kernel.reordered,
        block_size = 256,
        threads = 12,
        repetitions = 10,
        seed = 12345
    ):

        if type(float_type) is not FloatType or type(kernel) is not Kernel:
            raise TypeError('Use appropriate enums for spec')

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

    return pd.read_csv(results_csv_path)



if __name__ == '__main__':

    evaluate_specs(Specs([
        Spec(kernel = Kernel.naive),
        Spec(),
        Spec(k = 2048)
    ]))
