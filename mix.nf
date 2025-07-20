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
            def ff_min = row.ff_min
            def ff_max = row.ff_max
            def ff_number = row.ff_number
            def depth_min = row.depth_min
            def depth_max = row.depth_max
            def depth_number = row.depth_number
            def repeat = row.repeat
            return [meta, target_file, background_file, tsv_file, factor, ff_min, ff_max, ff_number, depth_min, depth_max, depth_number, repeat]
        }

    MIX(
        ch_samples,
        file("${workflow.projectDir}/bin/mix.py")
    )
}
