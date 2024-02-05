# regenieCondAnalysis


## test
移动到目录下

`cd regenieCond`

运行step1测试代码：
`./step1.sh -o tmp_step1 -T`

运行regenieCondAnalysis测试代码：
`./regenieCondAnalysis.sh -p ./test/test --phenoFile ./sup/regenie_qt.tsv --pheno ldl_a -t 20 --step1 tmp_step1/qt_step1_pred.list -o test/test_qt --defaultLOG10P 2 --defaultFREQ 1e-4 --exclude-mode 1`