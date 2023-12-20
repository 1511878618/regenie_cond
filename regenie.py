#!/usr/bin/env python
# -*- encoding: utf-8 -*-

'''
@Description:  Python version regenie
@Date     :2023/12/20
@Author      :Tingfeng Xu
@version      :1.0
'''

import argparse
import subprocess
import os
from pathlib import Path
import gzip
import shutil
from typing import Any, Union, List
import sys 

flags = ["lowmem", "ref-first", "bt", "qt"] # TODO: may add all flags
# comma_list_params = ["covarColList", "catCovarList", "phenoCol"]

# class BashCommand:
#     def __init__(self,verbose=True,  **args) -> None:
#         self.args = args
#         self.verbose = self.verbose
#         self.run_command = None
#     def check

class Regenie:
    # def __init__(self, pgen:str, phenoFile:str, pheno:Union[str, List], phenoCol:Union[str, List], keepFile:Union[str, List], regenie_mode:bool, covarFile:str, covarColList:Union[str, List], catCovarList:Union[str, List], maxCatLevels:int, bsize:int, out:str, minMAC:int, pred:str , **args):
    def __init__(self, verbose=True, **args):
        self.args = args
        self.verbose = verbose
        self.regenie_command=None 
        self.check_regenie()


    def check_regenie(self):
        if shutil.which("regenie") is None:
            msg = "regenie is not installed. Please install it before proceeding."
            sys.stderr.write(msg)
            sys.exit(1)
        else:
            if self.verbose:
                sys.stdout.write("regenie is installed at {}\n".format(shutil.which("regenie")))

    def build_regenie_command(self, **args):
        self.regenie_command = "regenie --step 2"
        for p, v in args.items():
            if v is None or v == []: # skip empty params
                continue 
            if p in flags: # flags
                if v:
                    self.regenie_command += f" --{p}"
            else: # params
                if isinstance(v, list):
                    v = ",".join(v) # TODO: may not work for --keep-files
                self.regenie_command += f" --{p} {v}"
        if self.verbose:
            sys.stdout.write(f"regenie command: {self.regenie_command}\n")

    def run_regenie_command(self):

        process = subprocess.Popen(self.regenie_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stderr = process.stderr
        for info in iter(process.stdout.readline, b''):
            sys.stdout.write(info.decode('utf-8'))

        process.wait()
        if process.returncode != 0:
            print(f"Error: {stderr.decode('utf-8')}")
            exit(1)
    def __call__(self) -> Any:
        self.build_regenie_command(**self.args)
        self.run_regenie_command()

class RegenieConditionalAnalysis:
    def __init__(self, **args):
        pass 

    def perform_conditional_analysis(self):
        self.output_path.mkdir(parents=True, exist_ok=True)
        for count in range(self.args.max_condsnp):
            print(f"-------------BEGIN OF epoch: {count} -------------")
            current_dir = self.output_path / f"cond_{count}"
            current_dir.mkdir(parents=True, exist_ok=True)

            # 构建并运行regenie命令
            regenie_command = self.construct_regenie_command(count, current_dir)
            self.run_regenie_command(regenie_command)

            # 处理regenie输出
            self.process_regenie_output(count, current_dir)

            print(f"-------------END OF epoch: {count} -------------")

    def construct_regenie_command(self, count, current_dir):
        command = self.regenie_command_base
        # 添加更多的命令参数
        # ...
        return command

    def process_regenie_output(self, count, current_dir):
        # 处理regenie输出文件
        # ...
        pass 

def get_parser():
    parser = argparse.ArgumentParser(description="Conditional Analysis By regenie")
    parser.add_argument('--pgen', required=True, help="plink pgen path")
    parser.add_argument('--phenoFile', required=True, help="phenotype file path")
    parser.add_argument('--phenoCol', required=False, nargs='+', help="phenotype column name(s) in the phenoFile")
    parser.add_argument('--keep', required=False, nargs='+', help="keep file path(s)")
    parser.add_argument('--bt', required=False, action="store_true", help="regenie mode bt")
    parser.add_argument('--qt', required=False, action="store_true", help="regenie mode qt")
    parser.add_argument('--ref-first', dest="ref-first",required=False, action="store_true", help="reference allele first")
    parser.add_argument('--covarFile', required=True, help="covariate file path")
    parser.add_argument('--covarColList', required=False, nargs='+', help="list of covariate column names")
    parser.add_argument('--catCovarList', required=False, nargs='+', help="list of categorical covariate names")
    parser.add_argument('--maxCatLevels', type=int, default=30, help="maximum categorical levels")
    parser.add_argument('--bsize', type=int, default=1000, help="block size for genotype blocks")
    parser.add_argument('--out', required=True, help="output prefix")
    parser.add_argument('--minMAC', type=int, default=5, help="minimum minor allele count for tested variants")
    parser.add_argument('--pred', required=True, help="file containing the list of predictions files from step 1")
    parser.add_argument("--threads", type=int, default=5,  help="number of threads")
    parser.add_argument("--lowmem", required=False, action="store_true", help="low memory mode")
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()
    args_dict = vars(args)
    print(args_dict)
    if args_dict["covarColList"] is None  and args_dict["catCovarList"] is None :
        # parser.error("At least one covariate is required")
        args_dict["covarColList"] = ["genotype_array", "inferred_sex", "age_visit", "PC1", "PC2", "PC3", "PC4", "PC5", "PC6", "PC7", "PC8", "PC9", "PC10", "assessment_center", "age_squared"]
        args_dict["catCovarList"] = ["genotype_array", "inferred_sex", "assessment_center"]
        args_dict["maxCatLevels"] = 30
        sys.stdout.write(f"will use default setting for covar, that is: {' '.join(args_dict['covarColList'])}")



    regenie = Regenie(**args_dict)
    regenie()

if __name__ == "__main__":
    main()
