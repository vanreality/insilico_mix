#!/usr/bin/env nextflow

include { MIX } from './modules/local/mix/main.nf'

workflow {
    ch_samples = Channel
        .fromPath(params.input_meta)
        .splitCsv(header: true, sep: '\t')
        .map { row ->
            def meta = [
                id: row.sample,
                label: row.label
            ]
            def target_file = file(row.target)
            def background_file = file(row.background)
            return [meta, target_file, background_file]
        }

    MIX(
        ch_samples,
        params.min_ff,
        params.max_ff,
        params.ff_step,
        params.min_depth,
        params.max_depth,
        params.depth_step,
        file(params.vcf),
        file("${workflow.projectDir}/bin/mix.py")
    )
}
