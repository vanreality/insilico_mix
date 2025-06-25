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

    def ffs = []
    java.math.BigDecimal.valueOf(params.min_ff).step(java.math.BigDecimal.valueOf(params.max_ff), java.math.BigDecimal.valueOf(params.ff_step)) {
        ffs.add(it.toFloat())
    }
    ch_ffs = Channel.fromList(ffs)

    def depths = (params.min_depth..params.max_depth).step(params.depth_step).collect()
    ch_depths = Channel.fromList(depths)

    ch_input = ch_samples
        .combine(ch_ffs)
        .combine(ch_depths)
        .map { meta, target_file, background_file, ff, depth ->
            def new_meta = meta + [ff: ff, depth: depth]
            return [ new_meta, target_file, background_file, ff, depth ]
        }

    ch_input.view()

    // MIX(
    //     ch_input,
    //     file(params.dmr_bed),
    //     file("${workflow.projectDir}/bin/mix.py")
    // )
}