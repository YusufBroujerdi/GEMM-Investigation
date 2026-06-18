import altair as alt
import copy
import dataframe_image as dfi
import itertools as it
import pandas as pd

import benchmarking as bm


optimal_block_sizes = {
    bm.Kernel.naive : 256,
    bm.Kernel.reordered : 256,
    bm.Kernel.tiled_a : 256,
    bm.Kernel.tiled_b : 256,
    bm.Kernel.multithreaded_a : 128,
    bm.Kernel.multithreaded_b : 128,
    bm.Kernel.multithreaded_c : 256
}

optimal_thread_counts = {
    bm.Kernel.multithreaded_a : 15,
    bm.Kernel.multithreaded_b : 15,
    bm.Kernel.multithreaded_c : 12,
    bm.Kernel.reordered : 8,
    bm.Kernel.naive : 8,
    bm.Kernel.tiled_a : 8,
    bm.Kernel.tiled_b : 8
}

def compare_index_strategies():

    df = bm.evaluate_specs(

        [
            bm.Spec(
                case_name = f'{k}_warmup',
                kernel = k,
                repetitions = 5,
                seed = 1000,
                m = 1024,
                n = 1024,
                k = 1024
            ) for k in [bm.Kernel.naive, bm.Kernel.multithreaded_c, bm.Kernel.reordered]
        ]

        + [
            bm.Spec(
                case_name = f'{k}_test_{i}',
                kernel = k,
                repetitions = 2,
                seed = i,
                m = 1024,
                n = 1024,
                k = 1024
            ) for (k, i) in it.product(
                [bm.Kernel.naive, bm.Kernel.multithreaded_c, bm.Kernel.reordered],
                range(0, 1)
            )
        ]
    )


    result = (
        df
        [~df['case_name'].str.contains(f'warmup')]
        .assign(kernel = lambda d: d['kernel'].map(lambda x: x.value))
        .groupby('kernel')
        .agg({
             'time_ms_mean' : 'mean',
             'gflops_per_second': 'mean',
             'time_ms_min' : 'min',
             'time_ms_max' : 'max'
        })
    )
    
    dfi.export(result.head(15).round(3), 'basic_test.png')

def get_optimal_tile_table():

    df = bm.evaluate_specs(

        [
            bm.Spec(
                case_name = f'{k}_warmup',
                kernel = k,
                repetitions = 5,
                seed = 1000
            ) for k in [bm.Kernel.tiled_a, bm.Kernel.tiled_b, bm.Kernel.reordered]
        ]

        + [
            bm.Spec(
                case_name = f'{k}_tile_{j}_test_{i}',
                kernel = k,
                block_size = j,
                repetitions = 5,
                seed = i
            ) for (k, j, i) in it.product(
                [bm.Kernel.tiled_a, bm.Kernel.tiled_b],
                [16 * 2 ** p for p in range(0, 9)],
                range(0, 5)
            )
        ]

    )

    tables = dict()
    for kernel in [bm.Kernel.tiled_a, bm.Kernel.tiled_b]:
        tables[kernel] = (
            df
            [df['kernel'] == kernel]
            [~df['case_name'].str.contains(f'warmup')]
            .groupby('block_size')
            .agg({
                 'time_ms_mean' : 'mean',
                 'gflops_per_second': 'mean',
                 'time_ms_min' : 'min',
                 'time_ms_max' : 'max'
            })
            .sort_values('block_size', ascending = False)
    )

    dfi.export(tables[bm.Kernel.tiled_a].head(15).round(3), 'tiled_a_optimal_params.png')
    dfi.export(tables[bm.Kernel.tiled_b].head(15).round(3), 'tiled_b_optimal_params.png')
    
    return tables[bm.Kernel.tiled_a], tables[bm.Kernel.tiled_b]
    


def get_tiling_performance_across_dim():


    def generate_specs(dim):

        return [
            bm.Spec(
                case_name = f'{k}_{dim}width_{j}_test_{i}',
                kernel = k,
                block_size = optimal_block_sizes[k],
                repetitions = 5,
                m = j if dim == 'm' else 2048,
                n = j if dim == 'n' else 2048,
                k = j if dim == 'k' else 2048,
                seed = i
            ) for (k, j, i) in it.product(
                [bm.Kernel.tiled_a, bm.Kernel.tiled_b, bm.Kernel.reordered],
                [2 ** p for p in range(0, (2048).bit_length())],
                range(0, 5)
            )
        ]
        

    def get_chart_data(width, dimension):

        return (
            df
            [df['case_name'].str.contains(f'_{width}_')]
            .assign(kernel = lambda d: d['kernel'].map(lambda x: x.value))
            .groupby([dimension, 'kernel'])
            .agg({
                    'time_ms_mean' : 'mean',
                    'gflops_per_second': 'mean',
                    'time_ms_min' : 'min',
                    'time_ms_max' : 'max'
            })
            .reset_index()
            .rename(columns = {dimension : width})
        )


    def generate_chart_dict(width, width_title, data):

        lines = {
            "mark" : {
                "type" : "line",
                "opacity" : 0.85
            },
            "encoding" : {
                "x" : {
                    "field" : width,
                    "type" : "quantitative",
                    "scale" : {"type" : "log", "base" : 2},
                    "axis" : {"title" : width_title}
                },
                "y" : {
                    "field" : "gflops_per_second",
                    "type" : "quantitative",
                    "axis" : {"title" : "GFLOP/s"}
                },
                "color" : {
                    "field" : "kernel",
                    "type" : "nominal"
                }
            }
        }

        points = copy.deepcopy(lines)
        points["mark"] = {
            "type" : "point",
            "opacity" : 0.65,
            "filled" : False
        }
        points["encoding"]["shape"] = {
            "field" : "kernel",
            "type" : "nominal",
            "legend" : None
        }

        return {
            "data" : alt.utils.data.to_values(data),
            "width" : 900,
            "height" : 500,
            "layer" : [lines, points]
        }


    df = bm.evaluate_specs(
        [
            bm.Spec(
                case_name = f'{k}_warmup',
                kernel = k,
                repetitions = 5,
                seed = 1000
            ) for k in [bm.Kernel.tiled_a, bm.Kernel.tiled_b, bm.Kernel.reordered]
        ]

        + generate_specs('m')
        + generate_specs('n')
        + generate_specs('k')
    )

    height_df = get_chart_data('mwidth', 'm')
    width_df = get_chart_data('nwidth', 'n')
    kwidth_df = get_chart_data('kwidth', 'k')

    height_chart = alt.Chart.from_dict(generate_chart_dict('mwidth', 'Height', height_df))
    width_chart = alt.Chart.from_dict(generate_chart_dict('nwidth', 'Width', width_df))
    kwidth_chart = alt.Chart.from_dict(generate_chart_dict('kwidth', 'K-Width', kwidth_df))

    height_chart.save('heights.png', scale_factor = 3)
    width_chart.save('widths.png', scale_factor = 3)
    kwidth_chart.save('kwidths.png', scale_factor = 3)
    
    return height_chart, width_chart, kwidth_chart



def get_optimal_thread_tile_table():

    df = bm.evaluate_specs(

        [
            bm.Spec(
                case_name = f'{k}_warmup',
                kernel = k,
                repetitions = 20,
                seed = 1000
            ) for k in [bm.Kernel.multithreaded_a, bm.Kernel.multithreaded_b]
        ]

        + [
            bm.Spec(
                case_name = f'{k}_tile_{j}_threadno_{t}_test_{i}',
                kernel = k,
                block_size = j,
                threads = t,
                repetitions = 10,
                seed = i
            ) for (k, t, j, i) in it.product(
                [bm.Kernel.multithreaded_a, bm.Kernel.multithreaded_b],
                range(8, 17),
                [16 * 2 ** p for p in range(1, 7)],
                range(0, 5)
            )
        ]
    )

    tables = dict()
    for kernel in [
        bm.Kernel.multithreaded_a, bm.Kernel.multithreaded_b
    ]:
        tables[kernel] = (
            df
            [df['kernel'] == kernel]
            [~df['case_name'].str.contains(f'warmup')]
            .groupby(['block_size', 'threads'])
            .agg({
                 'time_ms_mean' : 'mean',
                 'gflops_per_second': 'mean',
                 'time_ms_min' : 'min',
                 'time_ms_max' : 'max'
            })
            .sort_values('gflops_per_second', ascending = False)
        )

    dfi.export(tables[bm.Kernel.multithreaded_a].head(15).round(3), 'mt_a_optimal_params.png')
    dfi.export(tables[bm.Kernel.multithreaded_b].head(15).round(3), 'mt_b_optimal_params.png')

    return tables[bm.Kernel.multithreaded_a], tables[bm.Kernel.multithreaded_b]


    
def get_multithreaded_performance_across_dim():
    

    def generate_specs(dim):

        return [
            bm.Spec(
                case_name = f'{k}_{dim}width_{j}_test_{i}',
                kernel = k,
                block_size = optimal_block_sizes[k],
                threads = optimal_thread_counts[k],
                repetitions = 10,
                m = j if dim == 'm' else 2048,
                n = j if dim == 'n' else 2048,
                k = j if dim == 'k' else 2048,
                seed = i
            ) for (k, j, i) in it.product(
                [bm.Kernel.multithreaded_a, bm.Kernel.multithreaded_b],
                [2 ** p for p in range(0, (2048).bit_length())],
                range(0, 5)
            )
        ]
        

    def get_chart_data(width, dimension):

        return (
            df
            [df['case_name'].str.contains(f'_{width}_')]
            .assign(kernel = lambda d: d['kernel'].map(lambda x: x.value))
            .groupby([dimension, 'kernel'])
            .agg({
                 'time_ms_mean' : 'mean',
                 'gflops_per_second': 'mean',
                 'time_ms_min' : 'min',
                 'time_ms_max' : 'max'
            })
            .reset_index()
            .rename(columns = {dimension : width})
        )


    def generate_chart_dict(width, width_title, data):

        lines = {
            "mark" : {
                "type" : "line",
                "opacity" : 0.85
            },
            "encoding" : {
                "x" : {
                    "field" : width,
                    "type" : "quantitative",
                    "scale" : {"type" : "log", "base" : 2},
                    "axis" : {"title" : width_title}
                },
                "y" : {
                    "field" : "gflops_per_second",
                    "type" : "quantitative",
                    "axis" : {"title" : "GFLOP/s"}
                },
                "color" : {
                    "field" : "kernel",
                    "type" : "nominal"
                }
            }
        }

        points = copy.deepcopy(lines)
        points["mark"] = {
            "type" : "point",
            "opacity" : 0.65,
            "filled" : False
        }
        points["encoding"]["shape"] = {
            "field" : "kernel",
            "type" : "nominal",
            "legend" : None
        }

        return {
            "data" : alt.utils.data.to_values(data),
            "width" : 900,
            "height" : 500,
            "layer" : [lines, points]
        }  


    df = bm.evaluate_specs(
        [
            bm.Spec(
                case_name = f'{k}_warmup',
                kernel = k,
                repetitions = 20,
                seed = 1000
            ) for k in [bm.Kernel.multithreaded_a, bm.Kernel.multithreaded_b]
        ]

        + generate_specs('m')
        + generate_specs('n')
        + generate_specs('k')
    )

    height_df = get_chart_data('mwidth', 'm')
    width_df = get_chart_data('nwidth', 'n')
    kwidth_df = get_chart_data('kwidth', 'k')

    height_chart = alt.Chart.from_dict(generate_chart_dict('mwidth', 'Height', height_df))
    width_chart = alt.Chart.from_dict(generate_chart_dict('nwidth', 'Width', width_df))
    kwidth_chart = alt.Chart.from_dict(generate_chart_dict('kwidth', 'K-Width', kwidth_df))

    height_chart.save('heights.png', scale_factor = 3)
    width_chart.save('widths.png', scale_factor = 3)
    kwidth_chart.save('kwidths.png', scale_factor = 3)

    return height_chart, width_chart, kwidth_chart

    
if __name__ == '__main__':

    get_tiling_performance_across_dim()
