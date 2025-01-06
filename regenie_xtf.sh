#!/bin/bash
set -ex 
# step1 params 
hardCallPath=/pmaster/xutingfeng/dataset/ukb/dataset/snp/geneArray/ukb_cal_allChrs

# step2 params 
pgenPath=/pmaster/xutingfeng/dataset/ukb/dataset/snp/WES_IMP_GRCh38
# shared params 
threads=15
output=$2
memory=10G
# phenoFile=/pmaster/xutingfeng/dataset/ukb/dataset/regenie/union_v3/union_v3_INT_update.tsv
phenoFile=$1
covarFile=/pmaster/xutingfeng/dataset/ukb/dataset/regenie/union_v3_update/cov.regenie

covarColList=genotype_array,sex,age_visit,PC1,PC2,PC3,PC4,PC5,PC6,PC7,PC8,PC9,PC10,assessment_center,age_squared,BMI
catCovarList=genotype_array,sex,assessment_center
## people
extract=/pmaster/xutingfeng/dataset/ukb/dataset/snp/geneArray/qc_pass.snplist

keep1=/pmaster/xutingfeng/dataset/ukb/dataset/snp/geneArray/qc_pass.id
keep2=/pmaster/xutingfeng/dataset/ukb/dataset/regenie/union_v3_update/union_people.tsv




cat >&1 <<-EOF
Example code:
./run_regenie_xtf.sh phenofile outputfolder

EOF

mkdir -p ${output}
mkdir -p ${output}/step1
# run qt
if [ -f "${output}/step1/qt_step1_pred.list" ]; then
    echo "${output}/step1/qt_step1 already exists. Skipping step 1."
else
    exit 1
    regenie \
        --step 1 \
        --threads ${threads} \
        --bed ${hardCallPath} \
        --extract ${extract} \
        --keep  ${keep1}\
        --keep ${keep2} \
        --qt \
        --phenoFile ${phenoFile} \
        --covarFile ${covarFile} \
        --covarColList ${covarColList}\
        --catCovarList ${catCovarList} \
        --maxCatLevels 30 \
        --bsize 1000 \
        --out ${output}/step1/qt_step1
fi




for chr in {1..22}; do
    chr_output=${output}/step2/chr${chr}/
    mkdir -p ${chr_output}
    log_path=${output}/log
    mkdir -p ${log_path}
    chrpgen=${pgenPath}/ukb_wes_imp_chr${chr}
    sbatch -J "${chr}" -c ${threads} --mem=${memory} -o ${log_path}/chr${chr}_gwas.log  --wrap "regenie --step 2 --threads=${threads} --ref-first --pgen ${chrpgen} --phenoFile ${phenoFile} --keep ${keep2} --qt --covarFile ${covarFile} --covarColList ${covarColList} --catCovarList ${catCovarList} --maxCatLevels 30 --bsize 1000 --out ${chr_output} --minMAC 5 --pred ${output}/step1/qt_step1_pred.list "
done

# ./2_format.sh  ${output}/step2/  ${output}/summary/ ${threads}
# ./3_summary2pheweb.sh ${output}/summary/ ${output}/pheweb/ ${threads}