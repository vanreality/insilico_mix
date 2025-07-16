process MIX {
    input:
    tuple val(meta), path(target_file), path(background_file)
    val(min_ff)
    val(max_ff)
    val(ff_step)
    val(min_depth)
    val(max_depth)
    val(depth_step)
    path(vcf_file)
    path(script)
    
    output:
    tuple val(meta), path("*.parquet"), emit: mix_results
    path("*.log"), emit: log
    
    script:
    def args = task.ext.args ?: ''
    """
    python3 ${script} \\
        --target ${target_file} \\
        --background ${background_file} \\
        --ff-min ${min_ff} \\
        --ff-max ${max_ff} \\
        --ff-step ${ff_step} \\
        --depth-min ${min_depth} \\
        --depth-max ${max_depth} \\
        --depth-step ${depth_step} \\
        --vcf ${vcf_file} \\
        ${args} \
        > mix.log 2>&1
    """
}
