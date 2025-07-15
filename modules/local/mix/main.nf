process MIX {
    tag "${meta.id}_${meta.label}_${meta.ff}_${meta.depth}"
    
    input:
    tuple val(meta), path(target_file), path(background_file), val(ff), val(depth)
    path(vcf_file)
    path(script)
    
    output:
    tuple val(meta), path("*.parquet"), emit: mix_results
    
    script:
    def args = task.ext.args ?: ''
    """
    python3 ${script} \\
        --target ${target_file} \\
        --background ${background_file} \\
        --ff ${ff} \\
        --depth ${depth} \\
        --vcf ${vcf_file} \\
        --output ${meta.id}_${meta.label}_${ff.toString().replace('.', 'p')}_${depth} \\
        ${args}
    """
}
