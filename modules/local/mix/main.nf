process MIX {
    tag "${meta.id}"

    input:
    tuple val(meta), path(target_file), path(background_file), path(tsv_file), val(factor), val(min_ff), val(max_ff), val(ff_number), val(min_depth), val(max_depth), val(depth_number)
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
        --tsv ${tsv_file} \\
        --factor ${factor} \\
        --ff-min ${min_ff} \\
        --ff-max ${max_ff} \\
        --ff-number ${ff_number} \\
        --depth-min ${min_depth} \\
        --depth-max ${max_depth} \\
        --depth-number ${depth_number} \\
        --output-prefix ./ \\
        ${args} \
        > ${meta.id}.log 2>&1
    """
}
