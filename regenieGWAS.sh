#!/bin/bash
#########################################################
# Function : Regenie Step1                              #
# Platform : All Linux Based Platform                   #
# Version  : 1.4                                        #
# Date     : 2023-08-17                                 #
# Author   : Tingfeng Xu                                #
# Contact  : xutingfeng@big.ac.cn                       #
#########################################################

usage() {
    cat >&2 <<-EOF
使用方法: $(basename "$0") <option>

可用选项 <option> 包括:
    -p, --pfile <pfile>             plink pfile path, must be specified
    --phenoFile <phenoFile>         pheno file path, must be specified, like:/pmaster/xutingfeng/dataset/ukb/phenotype/regenie_qt.tsv
    --pheno <pheno>                  pheno name, must be specified
    -t, --threads <threads>         threads, default value is 20
    --step1 <step1>                 step1 pred file path, must be specified
    -o, --out <out>                 output prefix, default value is ./step2
    --bt                             bt mode, default is false, and be qt model
    --keep <keepFile>                   keep ID file, 用于限定使用的人群，如果不指定则使用全体样本
    --cov <covFile>                   cov file


示例:
    $(basename "$0") -p <pfile> --phenoFile <phenoFile> --pheno <pheno> -t <threads> --step1 <step1> -o <out>

    $(basename "$0")  -p /pmaster/xutingfeng/dataset/ukb/dataset/snp/test/SORT_pgen --phenoFile /pmaster/xutingfeng/dataset/ukb/phenotype/regenie_qt.tsv --pheno ldl_a -t 20 --step1 step1/qt_step1_pred.list -o test/test_qt

    parallel run qt: 
    parallel -q echo "sbatch -J '{1}_{2}_regenie' -c 10 --mem=15G -o log/%j_{1}_{2}.log --wrap 'regenieCondAnalysis.sh -p /pmaster/xutingfeng/dataset/ukb/dataset/snp/wgs/GRCh38/{1} --phenoFile /pmaster/xutingfeng/dataset/ukb/phenotype/regenie_qt.tsv --pheno {2} -t 10 --step1 ./step1/qt_step1_pred.list -o ./conditionalNew/{1}/{2} --exclude-mode 1 '" ::: APOB PCSK9 LDLR SORT1 ::: ldl_a apob |parallel

    parallel run bt:
    parallel -q echo "sbatch -J '{1}_{2}_regenie' -c 10 --mem=15G -o log/%j_{1}_{2}.log --wrap 'regenieCondAnalysis.sh -p /pmaster/xutingfeng/dataset/ukb/dataset/snp/wgs/GRCh38/{1} --phenoFile /pmaster/xutingfeng/dataset/ukb/phenotype/regenie_bt.tsv --pheno {2} -t 10 --step1 ./step1/bt_step1_pred.list -o ./conditionalNew/{1}/{2} --bt  --exclude-mode 1 '" ::: APOB PCSK9 LDLR SORT1 ::: cad mi |parallel 
注意:
    1. 所有参数都是必需的，除非另有说明。
    2. 默认参数值已在帮助文档中指定。

EOF
    exit 2
}

# 默认参数值
scriptPath=$(dirname "$0")

threads=20
outputPath="./step2"
regenie_mode="--qt"
keep_files=""
covarFile="${scriptPath}/regenie.cov"
# 定义必需的参数列表
required_params=("pgenPath" "phenoFile" "pheno" "predFile")

# set -e
# 解析命令行参数
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
    -p | --pfile)
        if [[ -z "$2" || "$2" == -* ]]; then
            echo "错误: --pfile 参数需要提供一个值" >&2
            exit 1
        fi
        pgenPath=$2
        echo "$1 $2"

        shift 2
        ;;
    --phenoFile)
        if [[ -z "$2" || "$2" == -* ]]; then
            echo "错误: --phenoFile 参数需要提供一个值" >&2
            exit 1
        fi
        phenoFile=$2
        echo "$1 $2"

        shift 2
        ;;
    --pheno)
        if [[ -z "$2" || "$2" == -* ]]; then
            echo "错误: --pheno 参数需要提供一个值" >&2
            exit 1
        fi
        pheno=$2
        shift 2
        ;;
    -t | --threads)
        if [[ -z "$2" || "$2" == -* ]]; then
            echo "错误: --threads 参数需要提供一个值" >&2
            exit 1
        fi
        threads=$2
        shift 2
        ;;
    --step1)
        if [[ -z "$2" || "$2" == -* ]]; then
            echo "错误: --step1 参数需要提供一个值" >&2
            exit 1
        fi
        predFile=$2
        echo "$1 $2"

        shift 2
        ;;
    -o | --out)
        if [[ -z "$2" || "$2" == -* ]]; then
            echo "错误: --out 参数需要提供一个值" >&2
            exit 1
        fi
        outputPath=$2
        shift 2
        ;;
    --bt)
        regenie_mode="--bt --firth --approx --pThresh 0.01"
        echo "$1"
        shift
        ;;
    --keep)
        if [[ -z "$2" || "$2" == -* ]]; then
            echo "错误: --keep 参数需要提供一个值" >&2
            exit 1
        fi

        keep_files="--keep $2"
        echo ${keep_files}
        shift 2
        ;;
    --cov)
        if [[ -z "$2" || "$2" == -* ]]; then
            echo "错误: --cov 参数需要提供一个值" >&2
            exit 1
        fi

        covarFile="$2"
        echo ${covarFile}
        shift 2
        ;;
    -h | --help)
        usage
        ;;
    *)
        echo "未知选项: $key" >&2
        usage
        exit 1
        ;;
    esac
done
# set -e
# 检查必需的参数是否提供
missing_params=()
for param in "${required_params[@]}"; do
    if [ -z "${!param}" ]; then
        missing_params+=("$param")
    fi
done

# 输出缺少的必需参数并退出
if [ ${#missing_params[@]} -gt 0 ]; then
    echo "缺少必需的参数: ${missing_params[*]}" >&2
    usage
    exit 1
fi

# 打印解析后的参数值
echo "pfile 参数值: $pgenPath"
echo "phenoFile 参数值: $phenoFile"
echo "pheno 参数值: $pheno"
echo "threads 参数值: $threads"
echo "step1 参数值: $predFile"
echo "out 参数值: $outputPath"
echo "regenie flag: $regenie_mode"
echo "covFile": ${covarFile}

mkdir -p ${outputPath}
regenie --step 2 \
    --threads=${threads} \
    --ref-first \
    --pgen ${pgenPath} \
    --phenoFile ${phenoFile} \
    --phenoCol ${pheno} \
    ${keep_files} \
    ${regenie_mode} \
    --covarFile ${covarFile} \
    --covarColList genotype_array,inferred_sex,age_visit,PC1,PC2,PC3,PC4,PC5,PC6,PC7,PC8,PC9,PC10,assessment_center,age_squared \
    --catCovarList genotype_array,inferred_sex,assessment_center \
    --maxCatLevels 30 \
    --bsize 1000 \
    --out ${outputPath}/ \
    --minMAC 1 \
    --pred ${predFile}
