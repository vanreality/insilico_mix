#!/usr/bin/env nextflow

include { MIX } from './modules/local/mix/main.nf'

workflow {
    ch_samples = Channel
        .fromPath(params.input_samplesheet)
        .splitCsv(header: true, sep: '\t')
        .map { row ->
            def meta = [id: row.sample]
            def target_file = file(row.target_pileup)
            def background_file = file(row.background_pileup)
            def tsv_file = file(row.tsv)
            def factor = row.factor
            def min_ff = row.min_ff
            def max_ff = row.max_ff
            def ff_number = row.ff_number
            def min_depth = row.min_depth
            def max_depth = row.max_depth
            def depth_number = row.depth_number
            return [meta, target_file, background_file, tsv_file, factor, min_ff, max_ff, ff_number, min_depth, max_depth, depth_number]
        }

    MIX(
        ch_samples,
        file("${workflow.projectDir}/bin/mix.py")
    )
}
