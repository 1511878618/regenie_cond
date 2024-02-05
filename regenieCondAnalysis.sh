#!/bin/bash
#########################################################
# Function : Conditional Analysis By regenie            #
# Platform : All Linux Based Platform                   #
# Version  : 1.5                                        #
# Date     : 2023-08-23                                 #
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
    --step2 <step2>                 step2 pred file path, optional
    -o, --out <out>                 output prefix, default value is ./conditionalAnalysis
    --exclude-mode <log10p_cutoff>  exclude mode, optional parameter, log10p_cutoff
    --max-condsnp  <value>           max cond snp, default value is 100
    --defaultLOG10P <value>         default log10p cutoff, default value is 6
    --defaultFREQ <value>           default freq cutoff, default value is 1e-2
    --bt                             bt mode, default is false, and be qt model
    --keep <keepFile>                   keep ID file, 用于限定使用的人群，如果不指定则使用全体样本
    --cov <covFile>                   cov file
    --condsnp-list <condsnpList>     cond snp list, 用于限定使用的cond snp

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
outputPath="./conditionalAnalysis"
maxcount=100
defaultLOG10P=6
defaultFREQ=1e-2
regenie_mode="--qt"
keep_files=""
covarFile="${scriptPath}/sup/regenie.cov"
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
    --step2)
        if [[ -z "$2" || "$2" == -* ]]; then
            echo "错误: --step2 参数需要提供一个值" >&2
            exit 1
        fi
        step2File=$2
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
    --exclude-mode)
        exclude_mode=true
        if [[ -z "$2" || "$2" == -* ]]; then
            echo "错误: --exclude-mode 参数需要提供一个值" >&2
            exit 1
        fi
        excludeLOG10PCUTOFF=$2
        echo "$1 $2"

        shift 2
        ;;
    --max-condsnp)
        if [[ -z "$2" || "$2" == -* ]]; then
            echo "错误: --max-count 参数需要提供一个值" >&2
            exit 1
        fi
        maxcount=$2
        echo "$1 $2"

        shift 2
        ;;
    --condsnp-list)
        if [[ -z "$2" || "$2" == -* ]]; then
            echo "错误: --condsnp-list 参数需要提供一个值" >&2
            exit 1
        fi
        condsnp_list=$2 
        echo "$1 $2"
        shift 2 
        ;; 
    --defaultLOG10P)
        if [[ -z "$2" || "$2" == -* ]]; then
            echo "错误: --defaultLOG10P 参数需要提供一个值" >&2
            exit 1
        fi
        defaultLOG10P=$2
        shift 2
        ;;
    --defaultFREQ)
        if [[ -z "$2" || "$2" == -* ]]; then
            echo "错误: --defaultFREQ 参数需要提供一个值" >&2
            exit 1
        fi
        defaultFREQ=$2
        echo "$1 $2"

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
echo "step2 参数值: $step2File"
echo "out 参数值: $outputPath"
echo "regenie flag: $regenie_mode"
echo "max-condsnp 参数值: $maxcount"
echo "defaultLOG10P 参数值: $defaultLOG10P"
echo "defaultFREQ 参数值: $defaultFREQ"
echo "covFile": ${covarFile}

if [ "$exclude_mode" = true ]; then
    echo "exclude-mode 参数值: $exclude_mode"
    echo "exclude-mode 参数值: $excludeLOG10PCUTOFF"
    echo "每一步得到的结果中LOG10P小于${excludeLOG10PCUTOFF}的SNP会被提出来作为下一步的排除"
    excludeSNPFile=${outputPath}/exclude.regenie

fi

# Export environments variable for regenie
alias regenie=${scriptPath}/bin/regenie_v3.2.6.gz_x86_64_Linux_mkl

mkdir -p ${outputPath}
leadingFile=${outputPath}/leading.regenie
for ((count = 0; count <= ${maxcount}; count++)); do
    # while [[ -n "${LOG10P}" && -n "${FREQ}" ]] && [ ${count} -le ${maxcount} ]; do
    # run regenie
    echo "-------------BEGIN OF epoch :${count} -------------"
    currentDir=${outputPath}/cond_${count}

    if [[ "${count}" -eq 0 ]]; then
        # Step1 run regenie step2
        mkdir -p ${currentDir}
        # 不存在step2File则直接运行step2
        if [[ -z "${step2File}" ]]; then
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
                --out ${currentDir}/ \
                --minMAC 1 \
                --pred ${predFile}
        else
            echo "存在step2File, cp file" # this is not good, TODO: change to ln or others
            if [[ ! -f "${step2File}" ]]; then
                echo "step2File:${step2File} 不存在"
                exit 1
            fi
            if [[ ${step2File} == *.gz ]]; then
                cp ${step2File} ${currentDir}/_${pheno}.regenie.gz
            else
                cp ${step2File} ${currentDir}/_${pheno}.regenie
            fi
        fi

    else
        mkdir -p ${currentDir}
        currentCondList=${currentDir}/cond.snplist
        if [[ ${condsnp_list}  &&  "${count}" -eq 1 ]]; then 
            # 采用新的condFile
            echo "使用新的condsnp_list:${condsnp_list}"
            currentCondList=${condsnp_list} # TODO: 排除pvar里面不包括的snp
        else
            # 提取当前需要用到的condList
            currentCondList=${currentDir}/cond.snplist
            tail -n +2 ${leadingFile} | awk '{print $3}' >${currentCondList}

        fi


        if [ "$exclude_mode" = true ]; then
            # 提取当前需要用到的excludeSNPList
            excludeSNPListCurrent=${currentDir}/exclude.snplist
            tail -n +2 ${excludeSNPFile} | awk '{print $3}' >${excludeSNPListCurrent}

            echo "exclude $(cat ${excludeSNPListCurrent} | wc -l) SNPs in ${count} conditionalAnalysis"

            # Step2.2 conditional analysis
            echo "running ${count} ........"
            regenie --step 2 \
                --threads=${threads} \
                --ref-first \
                --pgen ${pgenPath} \
                --phenoFile ${phenoFile} \
                --phenoCol ${pheno} \
                --condition-list ${currentCondList} \
                --exclude ${excludeSNPListCurrent} \
                ${keep_files} \
                ${regenie_mode} \
                --covarFile ${covarFile} \
                --covarColList genotype_array,inferred_sex,age_visit,PC1,PC2,PC3,PC4,PC5,PC6,PC7,PC8,PC9,PC10,assessment_center,age_squared \
                --catCovarList genotype_array,inferred_sex,assessment_center \
                --maxCatLevels 30 \
                --bsize 1000 \
                --out ${currentDir}/ \
                --minMAC 1 \
                --pred ${predFile}
            # >/dev/null

        else
            # Step2.2 conditional analysis
            # echo "running ${count} ........"
            regenie --step 2 \
                --threads=${threads} \
                --ref-first \
                --pgen ${pgenPath} \
                --phenoFile ${phenoFile} \
                --phenoCol ${pheno} \
                --condition-list ${currentCondList} \
                ${keep_files} \
                ${regenie_mode} \
                --covarFile ${covarFile} \
                --covarColList genotype_array,inferred_sex,age_visit,PC1,PC2,PC3,PC4,PC5,PC6,PC7,PC8,PC9,PC10,assessment_center,age_squared \
                --catCovarList genotype_array,inferred_sex,assessment_center \
                --maxCatLevels 30 \
                --bsize 1000 \
                --out ${currentDir}/ \
                --minMAC 1 \
                --pred ${predFile}
            # >/dev/null

        fi

    fi
    # 对输出处理

    currentRegenieOutPut=${currentDir}/_${pheno}.regenie

    if [[ -s ${currentRegenieOutPut} ]]; then
        gzip ${currentRegenieOutPut}
    fi

    # 写入header

    if [[ "${count}" -eq 0 ]]; then # 写入header
        read currentSNP FREQ LOG10P <<<"$(zcat ${currentRegenieOutPut}.gz | awk 'NR==1{print;next}NR>=2{print | "sort -k12gr"}' | awk -v freqCUTOFF=${defaultFREQ} -v LOG10PCUTOFF=${defaultLOG10P} 'NR==1{print $0, "FAILDTIME";next}{if ($6 > freqCUTOFF && $12 >LOG10PCUTOFF){print $0, "leading";exit}}' | tee ${leadingFile} | awk 'BEGIN{OFS=" "}NR==2{print $3, $6, $12}')"
    else # 不写入header
        read currentSNP FREQ LOG10P <<<"$(zcat ${currentRegenieOutPut}.gz | awk 'NR>=2{print | "sort -k12gr"}' | awk -v freqCUTOFF=${defaultFREQ} -v LOG10PCUTOFF=${defaultLOG10P} '{if ($6 > freqCUTOFF && $12 >LOG10PCUTOFF){print $0, "leading";exit}}' | tee -a ${leadingFile} | awk 'BEGIN{OFS=" "}{print $3, $6, $12}')"
    fi
    # 提取排除 的SNP
    if [[ "$exclude_mode" = true ]]; then
        if [[ "${count}" -eq 0 ]]; then
            zcat "${currentRegenieOutPut}".gz | awk -v excludeLOG10PCUTOFF="${excludeLOG10PCUTOFF}" 'NR==1{print $0, "FAILDTIME";next}$12 <excludeLOG10PCUTOFF {print $0, "0"}' >${excludeSNPFile} # 提取对应的最后一次所有的信息，最后一列为FAILDTIME；说明是在那一次被过滤掉的。
        else
            # 如果exclude_mode open 则提取当前的excludeSNPList
            zcat "${currentRegenieOutPut}".gz | awk -v excludeLOG10PCUTOFF="${excludeLOG10PCUTOFF}" -v currentCount=${count} 'NR==1{next}$12 <excludeLOG10PCUTOFF {print $0, currentCount}' >>"${excludeSNPFile}" # 提取对应的最后一次所有的信息
        fi
    fi
    # 更新计数器
    if [[ -n "${LOG10P}" && -n "${FREQ}" ]]; then
        echo -e "epoch\tID\tFREQ\tLOG10P\ncond_${count}\t${currentSNP}\t${FREQ}\t${LOG10P}" | column -t
        echo "-------------END OF epoch:${count}  -------------"

    else
        echo "No SNP passed the filter at epoch ${count}!!!!!"
        echo "-------------END OF epoch:${count}  -------------"

        break
    fi
done
leadingSNPList=${outputPath}/leading.snplist
tail -n +2 ${leadingFile} | awk '{print $3}' >${leadingSNPList}

finalFile=${outputPath}/${pheno}.cond.regeine
# 如果exclude_mode open 则提取当前的excludeSNPList
if [ "$exclude_mode" = true ]; then
    # excludeSNPList_current=${outputPath}/exclude.snplist
    finalKeepFile=${outputPath}/keep.regenie

    zcat "${currentRegenieOutPut}".gz | awk -v excludeLOG10PCUTOFF="${excludeLOG10PCUTOFF}" -v currentCount=${count} 'NR==1{print $0, "FAILDTIME";next}$12 >=excludeLOG10PCUTOFF {print $0, "keep"}' >${finalKeepFile}
    # merge all
    cat ${leadingFile} <(tail -n +2 ${finalKeepFile}) <(tail -n +2 ${excludeSNPFile}) >${finalFile}

else
    cat ${leadingFile} <(zcat "${currentRegenieOutPut}".gz | awk 'NR>=2{print $0, "keep"}') >${finalFile}
fi

echo "Finished!"
