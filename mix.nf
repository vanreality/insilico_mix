#!/usr/bin/env nextflow

include { MIX } from './modules/local/mix/main.nf'

workflow {
    ch_samples = Channel
        .fromPath(params.input_samplesheet)
        .splitCsv(header: true)
        .map { row ->
            def meta = [id: row.out_prefix]
            def target_file = file(row.target_pileup)
            def background_file = file(row.background_pileup)
            def pl_ff = row.pl_ff
            def cfdna_ff = row.cfdna_ff
            def mix_ff = row.mix_ff
            def replicate = row.replicate
            return [meta, target_file, background_file, pl_ff, cfdna_ff, mix_ff, replicate]
        }

    MIX(
        ch_samples,
        file("${workflow.projectDir}/bin/mix_pileup.py")
    )
}
