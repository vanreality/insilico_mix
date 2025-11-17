process MIX {
    tag "${meta.id}"

    input:
    tuple val(meta), path(target_file), path(background_file), val(pl_ff), val(cfdna_ff), val(mix_ff), val(replicate)
    path(script)
    
    output:
    tuple val(meta), path("*_pileup.tsv.gz"), emit: mix_results
    path("*.log"), emit: log
    
    script:
    def args = task.ext.args ?: ''
    """
    python3 ${script} \\
        --target ${target_file} \\
        --background ${background_file} \\
        --target_ff ${pl_ff} \\
        --background_ff ${cfdna_ff} \\
        --mix_ff ${mix_ff} \\
        --output_prefix ${meta.id} \\
        ${args} \
        > ${meta.id}.log 2>&1
    """
}
