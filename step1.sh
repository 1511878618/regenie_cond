#!/bin/bash
#########################################################
# Function : Regenie Step1 By regenie                   #
# Platform : All Linux Based Platform                   #
# Version  : 1.0                                        #
# Date     : 2023-08-07                                 #
# Author   : Tingfeng Xu                                #
# Contact  : xutingfeng@big.ac.cn                       #
#########################################################
usage() {
    cat >&2 <<-EOF
使用方法: $(basename "$0") <option>

可用选项 <option> 包括:
    -s, --hardCallPath <hardCallPath>  hardCallPath
    -t, --threads <threads>  threads
    -o, --output <output>  output
    -p, --PhenoPath <PhenoPath>  PhenoPath, this should contain regeine_qt.tsv or regenie_bt.tsv
    --cov <covarFile>  covarFile
    -T, --test flag, test mode

示例:
    $(basename "$0") -p <pfile> --phenoFile <phenoFile> --pheno <pheno> -t 
如果测试：
    ./step1.sh -o tmp_step1 -T
注意:
    1. 所有参数都是必需的，除非另有说明。
    2. 默认参数值已在帮助文档中指定。

EOF
    exit 2
}

#!/bin/bash

# Default values
threads=4
output="output_folder"
test_mode=false

# Parse command line options
while [[ $# -gt 0 ]]; do
    case "$1" in
    -s | --hardCallPath)
        hardCallPath="$2"
        shift 2
        ;;
    -t | --threads)
        threads="$2"
        shift 2
        ;;
    -o | --output)
        output="$2"
        shift 2
        ;;
    -p | --PhenoPath)
        PhenoPath="$2"
        shift 2
        ;;
    -T | --test)
        test_mode=true
        shift
        ;;

    --cov)
        if [[ -z "$2" || "$2" == -* ]]; then
            echo "错误: --cov 参数需要提供一个值" >&2
            exit 1
        fi

        covarFile="$2"
        shift 2
        ;;
    -h | --help)
        usage
        ;;
    *)
        echo "Unknown option: $1"
        exit 1
        ;;
    esac
done

if [ "$test_mode" = true ]; then
    echo "Test mode is on."
    echo "Using default values:"
    scriptPath=$(dirname "$0")
    threads=4
    output=${output}
    PhenoPath=${scriptPath}/sup
    covarFile=${scriptPath}/sup/regenie.cov
    hardCallPath=${scriptPath}/test/hardCall_test
    if [ -z "$output" ]; then
        echo "Error: Missing required options."
        exit 1
    fi

    echo "threads: $threads"
    echo "output: $output"
    echo "PhenoPath: $PhenoPath"
    echo "covarFile: $covarFile"

else
    # Validate required options
    if [ -z "$hardCallPath" ] || [ -z "$PhenoPath" ] || [ -z "$covarFile" ] || [ -z "$output" ]; then
        echo "Error: Missing required options."
        exit 1
    fi
    # Print parsed options
    echo "hardCallPath: $hardCallPath"
    echo "threads: $threads"
    echo "output: $output"
    echo "PhenoPath: $PhenoPath"
    echo "covarFile: $covarFile"
fi

mkdir -p ${output}

# sbatch -J "regene_qt" -c ${threads} --mem=40G -o qt.sbatch.log
# --wrap "
regenie \
    --step 1 \
    --threads ${threads} \
    --bed ${hardCallPath} \
    --extract /pmaster/xutingfeng/dataset/ukb/dataset/snp/geneArray/qc_pass.snplist \
    --keep /pmaster/xutingfeng/dataset/ukb/dataset/snp/geneArray/qc_pass.id \
    --keep /pmaster/xutingfeng/dataset/ukb/phenotype/white.tsv \
    --qt \
    --phenoFile ${PhenoPath}/regenie_qt.tsv \
    --covarFile ${covarFile} \
    --covarColList genotype_array,inferred_sex,age_visit,PC1,PC2,PC3,PC4,PC5,PC6,PC7,PC8,PC9,PC10,assessment_center,age_squared \
    --catCovarList genotype_array,inferred_sex,assessment_center \
    --maxCatLevels 30 \
    --bsize 1000 \
    --out ${output}/qt_step1
# "

# run bt step1
# sbatch -J "regene_bt" -c ${threads} --mem=40G -o bt.sbatch.log \
#     --wrap "
regenie \
    --step 1 \
    --threads ${threads} \
    --bed ${hardCallPath} \
    --extract /pmaster/xutingfeng/dataset/ukb/dataset/snp/geneArray/qc_pass.snplist \
    --keep /pmaster/xutingfeng/dataset/ukb/dataset/snp/geneArray/qc_pass.id \
    --keep /pmaster/xutingfeng/dataset/ukb/phenotype/white.tsv \
    --bt \
    --phenoFile ${PhenoPath}/regenie_bt.tsv \
    --covarFile ${covarFile} \
    --covarColList genotype_array,inferred_sex,age_visit,PC1,PC2,PC3,PC4,PC5,PC6,PC7,PC8,PC9,PC10,assessment_center,age_squared \
    --catCovarList genotype_array,inferred_sex,assessment_center \
    --maxCatLevels 30 \
    --firth \
    --approx \
    --pThresh 0.01 \
    --bsize 1000 \
    --out ${output}/bt_step1
# "
