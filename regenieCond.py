#!/usr/bin/env python
# -*- encoding: utf-8 -*-

"""
@Description:  Python version regenie
@Date     :2023/12/20
@Author      :Tingfeng Xu
@version      :1.0
"""

import argparse
import subprocess

from pathlib import Path
import shutil
from typing import Union, List, Dict, Any, Tuple
import textwrap

import sys

flags = ["lowmem", "ref-first", "bt", "qt"]  # TODO: may add all flags
default_exclude_log10p_cutoff = 1


def filter_regenie(
    regenie_summary_path: Union[str, Path],
    log10p_cutoff: float = 6,
    freq_cutoff: float = 1e-2,
    exclude_log10p_cutoff=None,
) -> List[Dict[str, Any]]:
    """
    Filter the regenie summary data based on log10p and frequency cutoffs.

    Args:
        regenie_summary_path (Union[str, Path]): The path to the regenie summary file.
        log10p_cutoff (float, optional): The log10p cutoff value. Defaults to 6.
        freq_cutoff (float, optional): The frequency cutoff value. Defaults to 1e-2.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing the filtered rows.

    """
    passed_rows = []
    # activate only for exclude mode
    if exclude_log10p_cutoff is not None:
        exclude_ids = []

    line_idx = 0
    with open(regenie_summary_path, "r") as f:
        for line in f:
            if line_idx == 0:
                header = line.strip().split()
                # continue
            else:
                line = line.strip().split()
                line_dict = dict(zip(header, line))
                log10p = float(line_dict["LOG10P"])
                freq = float(line_dict["A1FREQ"])

                if log10p > log10p_cutoff and freq <= freq_cutoff:
                    passed_rows.append(line_dict)
                if exclude_log10p_cutoff is not None:
                    if log10p <= exclude_log10p_cutoff:
                        exclude_ids.append(line_dict)

            line_idx += 1

    passed_rows = list(
        sorted(passed_rows, key=lambda x: float(x["LOG10P"]), reverse=True)
    )
    passed_num = len(passed_rows)
    sys.stdout.write(f"passed {passed_num} snps\n")
    passed_snp_dict = [passed_rows[0]] if passed_num>=1 else []  # besure to return a list[dict]
    if exclude_log10p_cutoff is not None:
        exclude_num = len(exclude_ids)
        sys.stdout.write(f"exclude {exclude_num} snps\n")

        if exclude_num ==0:
            exclude_ids = None

        return passed_snp_dict, exclude_ids
    else:
        return passed_snp_dict, None

def check_sorted_snpid(snpid:str):
    if ":" in snpid:
        chr, pos, a1, a2 = snpid.split(":")
        return ':'.join([chr, pos, *list(sorted([a1, a2]))])
    else:
        return None

def extract_snp_from_regenie_summary(
    snp_id_list: List[str], regenie_summary_path: Union[str, Path]
) -> List[Dict[str, str]]:
    """
    Extracts SNPs from a regenie summary file based on a given list of SNP IDs.

    Args:
        snp_id_list (List[str]): A list of SNP IDs to extract.
        regenie_summary_path (Union[str, Path]): The path to the regenie summary file.

    Returns:
        List[Dict[str, str]]: A list of dictionaries containing the extracted SNP information.

    Raises:
        FileNotFoundError: If the regenie summary file does not exist.

    Example:
        snp_ids = ["rs123", "rs456", "rs789"]
        summary_path = "/path/to/regenie_summary.txt"
        extracted_snps = extract_snp_from_regenie_summary(snp_ids, summary_path)
    """
    extracted_snp_list = []
    line_idx = 0
    with open(regenie_summary_path, "r") as f:
        for line in f:
            if line_idx == 0:
                header = line.strip().split()
                
            else:
                line = line.strip().split()
                line_dict = dict(zip(header, line))
                snp_id = line_dict["ID"]
                if snp_id in snp_id_list:
                    extracted_snp_list.append(line_dict)
                else: # check is : sep and sorted?
                    sorted_snp_id = check_sorted_snpid(snp_id)
                    if sorted_snp_id is not None and sorted_snp_id in snp_id_list: # sorted and matched
                        extracted_snp_list.append(line_dict)
                        sys.stdout.write(f"Warning: {snp_id} is not sorted, but passed\n with {sorted_snp_id}")
                        extracted_snp_list.append(line_dict)
             
            line_idx += 1
    if len(extracted_snp_list) != len(snp_id_list):
        for snp_id in snp_id_list:
            if snp_id not in [x["ID"] for x in extracted_snp_list]:
                sys.stdout.write(
                    f"Error: {snp_id} not in regenie summary file, please check your snp list\n"
                )
        sys.stdout.write(
            f"succeed extract {len(extracted_snp_list)} snps from regenie summary file while passed snp list has {len(snp_id_list)} snps\n "
        )
    return extracted_snp_list


class Regenie:
    """
    Regenie class represents a wrapper for the regenie command-line tool.

    Args:
        verbose (bool, optional): Whether to display verbose output. Defaults to True.
        **args: Additional arguments to be passed to the regenie command.

    Attributes:
        args (dict): Additional arguments to be passed to the regenie command.
        verbose (bool): Whether to display verbose output.
        regenie_command (str): The constructed regenie command.
    """

    def __init__(self, verbose=True, **args):
        self.args = args
        self.verbose = verbose
        self.regenie_command = None
        self.check_regenie()

    def check_regenie(self):
        """
        Checks if regenie is installed.

        Raises:
            SystemExit: If regenie is not installed.
        """
        if shutil.which("regenie") is None:
            msg = "regenie is not installed. Please install it before proceeding."
            sys.stderr.write(msg)
            sys.exit(1)
        else:
            if self.verbose:
                sys.stdout.write(
                    "regenie is installed at {}\n".format(shutil.which("regenie"))
                )

    def build_regenie_command(self, **args):
        """
        Builds the regenie command based on the provided arguments.

        Args:
            **args: Additional arguments to be passed to the regenie command.
        """
        self.regenie_command = "regenie --step 2"
        for p, v in args.items():
            if v is None or v == []:  # skip empty params
                continue
            if p in flags:  # flags
                if v:
                    self.regenie_command += f" --{p}"
            else:  # params
                if isinstance(v, list):
                    v = ",".join(v)  # TODO: may not work for --keep-files
                self.regenie_command += f" --{p} {v}"
        if self.verbose:
            sys.stdout.write(f"regenie command: {self.regenie_command}\n")

    def run_regenie_command(self):
        """
        Runs the regenie command.

        Raises:
            SystemExit: If the regenie command returns a non-zero exit code.
        """
        process = subprocess.Popen(
            self.regenie_command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        stderr = process.stderr
        for info in iter(process.stdout.readline, b""):
            sys.stdout.write(info.decode("utf-8"))

        process.wait()
        if process.returncode != 0:
            print(f"Error: {stderr.decode('utf-8')}")
            exit(1)

    def __call__(self) -> Any:
        """
        Executes the Regenie object.

        Returns:
            Any: The result of executing the regenie command.
        """
        self.build_regenie_command(**self.args)
        self.run_regenie_command()


class RegenieConditionalAnalysis:
    def __init__(self, args):

        self.args = args

        self.parse_args()

    def parse_args(self):
            """
            解析命令行参数，并设置相关参数。

            Args:
                None

            Returns:
                None
            """
            
            args_dict = self.args
            # regenie Cond params parse
            regenieCond_params = [
                "summary",
                "outputFolder",
                "disable-exclude-mode",
                "max-condsnp",
                "condsnp-path",
                "condsnp-list",
                "defaultLOG10P",
                "defaultFREQ",
            ]
            print(args_dict)
            regenieCond_args = {
                k: args_dict.pop(k) for k in regenieCond_params if k in args_dict.keys()
            }
            for k, v in regenieCond_args.items():
                if v is not None:
                    sys.stdout.write(f"regenieCond param: --{k} {v}\n")

            if (
                regenieCond_args["condsnp-path"] is None
                and regenieCond_args["condsnp-list"] is None
            ):
                sys.stdout.write(
                    "will use condsnp list by gwas result instead of any prior condsnp list file\n"
                )

            if regenieCond_args["summary"] is not None:
                sys.stdout.write(
                    f"will use summary: {regenieCond_args['summary']} and wont run step2 again\n"
                )

            # regenie default setting
            if args_dict["covarColList"] is None and args_dict["catCovarList"] is None:
                # parser.error("At least one covariate is required")
                args_dict["covarColList"] = [
                    "genotype_array",
                    "inferred_sex",
                    "age_visit",
                    "PC1",
                    "PC2",
                    "PC3",
                    "PC4",
                    "PC5",
                    "PC6",
                    "PC7",
                    "PC8",
                    "PC9",
                    "PC10",
                    "assessment_center",
                    "age_squared",
                ]
                args_dict["catCovarList"] = [
                    "genotype_array",
                    "inferred_sex",
                    "assessment_center",
                ]
                args_dict["maxCatLevels"] = 30
                sys.stdout.write(
                    f"will use default setting for covar, that is: {' '.join(args_dict['covarColList'])}"
                )
            # conditional analysis should use specific pheno, so if phenoCol is None, raise warning
            if args_dict["phenoCol"] is None:
                sys.stderr.write("Error: phenoCol is None, please specify phenoCol\n")
                sys.exit(1)
            elif isinstance(args_dict["phenoCol"], list):
                args_dict["phenoCol"] = args_dict["phenoCol"][0]

            # set args
            self.cond_args = regenieCond_args
            self.regenie_default_args = args_dict

    def perform_conditional_analysis(self):
        
        # extract params
        cond_args = self.cond_args
        regenie_default_args = self.regenie_default_args
        
        # mkdir output folder
        output_path = Path(cond_args["outputFolder"])
        if not output_path.exists():
            sys.stdout.write(f"mkdir {output_path}\n")
            output_path.mkdir(parents=True, exist_ok=True)

        # already haved cond snp list
        already_haved_cond_snp_list = []
        used_cond_snp_list_path = output_path / "used_cond_snp_list.csv"

        # exclude_snp_path
        exclude_snp_path = output_path / "exclude_snp.csv"
        # final result file
        final_result_path = output_path / "final_result.regenieCond"

        iter_count = 0
        while True:
            print(f"-------------BEGIN OF epoch: {iter_count} -------------")
            current_dir = output_path / f"cond_{iter_count}"
            # mkdir current dir
            if not current_dir.exists():
                sys.stdout.write(f"mkdir {current_dir}\n")
                current_dir.mkdir(parents=True, exist_ok=True)

            # run regenie
            if iter_count == 0:  # run step 2
                if cond_args["summary"]:
                    current_regenie_output_file = Path(cond_args["summary"])
                    if not current_regenie_output_file.exists():
                        sys.stderr.write(
                            f"Error: {current_regenie_output_file} not exists"
                        )
                        sys.exit(1)
                else:
                    # 构建并运行regenie命令
                    current_regenie_args = regenie_default_args.copy()
                    current_regenie_args["out"] = str(current_dir) + "/"

                    regenie_engine = Regenie(**current_regenie_args)
                    regenie_engine()  # run

                    # 处理regenie输出, file is named as _${pheno}.regenie
                    current_regenie_output_file = (
                        current_dir / f"_{current_regenie_args['phenoCol']}.regenie"
                    )
            else:
                # 构建并运行regenie命令
                current_regenie_args = regenie_default_args.copy()
                current_regenie_args["out"] = str(current_dir) + "/"

                # cond_params pass

            # parse results

            if iter_count == 0:
                # the first iteration
                if (
                    cond_args["condsnp-path"] is None
                    and cond_args["condsnp-list"] is None
                ):
                    # use gwas result as condsnp list
                    condsnp_list, exclude_snp_list = filter_regenie(
                        current_regenie_output_file,
                        cond_args["defaultLOG10P"],
                        cond_args["defaultFREQ"],
                        exclude_log10p_cutoff=default_exclude_log10p_cutoff
                        if cond_args["disable-exclude-mode"]
                        else None,
                    )
                else:
                    if (
                        cond_args["condsnp-path"] is not None
                        and cond_args["condsnp-list"] is None
                    ):
                        condsnp_id_list = [
                            x.strip() for x in open(cond_args["condsnp-path"], "r")
                        ]
                    elif (
                        cond_args["condsnp-path"] is None
                        and cond_args["condsnp-list"] is not None
                    ):
                        condsnp_id_list = cond_args["condsnp-list"]
                    else:
                        sys.stdout.write(
                            "both condsnp-path and condsnp-list are specified, will concat them and keep unique"
                        )
                        condsnp_id_list = [
                            x.strip() for x in open(cond_args["condsnp-path"], "r")
                        ] + cond_args["condsnp-list"]
                        condsnp_id_list = list(set(condsnp_id_list))

                    # check snp isin summary or step2 result
                    condsnp_list = extract_snp_from_regenie_summary(
                        condsnp_id_list, current_regenie_output_file
                    )
            else:
                # not the first iteration
                # extract leading from current_regenie_output_file
                condsnp_list, exclude_snp_list = filter_regenie(
                    current_regenie_output_file,
                    cond_args["defaultLOG10P"],
                    cond_args["defaultFREQ"],
                    exclude_log10p_cutoff=default_exclude_log10p_cutoff if cond_args["disable-exclude-mode"] else None,
                )

            if len(condsnp_list) == 0:
                sys.stdout.write(
                    f"no snp passed log10p and freq cutoff, will stop\n"
                )
                break

            # update already_haved_cond_snp_list with current condsnp_list
            already_haved_cond_snp_list.append(condsnp_list)

            # save condsnp_list to file, appendix mode
            with open(used_cond_snp_list_path, "a") as f:
                for snp_dict in condsnp_list:
                    snp_id = snp_dict["ID"]
                    f.write(f"{snp_id}\n")

            # update leadning snp to stdout
            for snp_dict in condsnp_list:
                sys.stdout.write("\t".join(snp_dict.keys()) + "\n")
                sys.stdout.write("\t".join(snp_dict.values())+ "\n")

            if (
                cond_args["max-condsnp"] != -1
                and iter_count >= cond_args["max-condsnp"]
            ):
                sys.stdout.write(f"max condsnp limit reached, will stop\n")
                break

            # update leading into final result file
            with open(final_result_path, "a") as f:
                if iter_count == 0:
                    header = "\t".join(list(condsnp_list[0].keys()) + ["FAILDTIME"])
                    f.write(header + "\n")
                for snp_dict in condsnp_list:
                    snp_dict["FAILDTIME"] = -1 
                    line = "\t".join(snp_dict.values())
                    f.write(line + "\n")

            # exclude mode files update 
            if exclude_snp_list is not None and not cond_args["disable-exclude-mode"]:
                # update exclude snp list
                with open(exclude_snp_path, "a") as f:
                    for snp_dict in exclude_snp_list:
                        snp_id = snp_dict["ID"]
                        f.write(f"{snp_id}\n")
                # update exclude snp to final result file
                with open(final_result_path, "a") as f:
                    for snp_dict in exclude_snp_list:
                        snp_dict["FAILDTIME"] = str(iter_count)
                        line = "\t".join(snp_dict.values())
                        f.write(line + "\n")

                
                    
        # end
        sys.stdout.write(f"-------------END OF epoch: {iter_count} -------------\n")
        # update final result file
        line_idx = 0
        with open(current_regenie_output_file, "r") as f:
            for line in f:
                if line_idx == 0 and iter_count == 0: # only when the first iteration is breaked 
                    header = "\t".join(line.strip().split() + ["FAILDTIME"])
                    to_write = header
                else:
                    line = line.strip().split() + [str(iter_count)]
                    to_write = "\t".join(line)
                with open(final_result_path, "a") as h:
                    h.write(to_write + "\n")
                line_idx += 1

                

        sys.stdout.write(f"result path: {str(final_result_path)}\n")

def get_parser():
    parser = argparse.ArgumentParser(description="Conditional Analysis By regenie")

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(
            """
        %prog Conditional Analysis By regenie
        @Author: xutingfeng@big.ac.cn
        Version: 1.0
        Example Code:
        1. if no --covarColList and --catCovarList, will use default setting or add this 
        
        regenieCond.py --pgen test/test --phenoFile sup/regenie_bt.tsv --bt  --phenoCol cad --covarFile sup/regenie.cov --bsize 1000 --pred tmp_step1/bt_step1_pred.list  --ref-first 

        2. if no --summary, will run step2 again, and will use default setting or add this

        3. if --summary, will use this summary file and wont run step2 again, and will use default setting or add this

        4. if use --condsnp-path or --condsnp-list, will use this condsnp list, and will use default setting or add this

        if no --condsnp-path and --condsnp-list, will use gwas result as condsnp list, and will use default setting or add this


        """
        ),
    )
    parser.add_argument("--pgen", required=True, help="plink pgen path")
    parser.add_argument("--phenoFile", required=True, help="phenotype file path")
    parser.add_argument(
        "--phenoCol",
        required=False,
        nargs="+",
        help="phenotype column name(s) in the phenoFile",
    )
    parser.add_argument("--keep", required=False, nargs="+", help="keep file path(s)")
    parser.add_argument(
        "--bt", required=False, action="store_true", help="regenie mode bt"
    )
    parser.add_argument(
        "--qt", required=False, action="store_true", help="regenie mode qt"
    )
    parser.add_argument(
        "--ref-first",
        dest="ref-first",
        required=False,
        action="store_true",
        help="reference allele first",
    )
    parser.add_argument("--covarFile", required=True, help="covariate file path")
    parser.add_argument(
        "--covarColList",
        required=False,
        nargs="+",
        help="list of covariate column names",
    )
    parser.add_argument(
        "--catCovarList",
        required=False,
        nargs="+",
        help="list of categorical covariate names",
    )
    parser.add_argument(
        "--maxCatLevels", type=int, default=30, help="maximum categorical levels"
    )
    parser.add_argument(
        "--bsize", type=int, default=1000, help="block size for genotype blocks"
    )
    # parser.add_argument('--out', required=True, help="output prefix")
    parser.add_argument(
        "--minMAC",
        type=int,
        default=5,
        help="minimum minor allele count for tested variants",
    )
    parser.add_argument(
        "--pred",
        required=True,
        help="file containing the list of predictions files from step 1",
    )
    parser.add_argument("--threads", type=int, default=5, help="number of threads")
    parser.add_argument(
        "--lowmem", required=False, action="store_true", help="low memory mode"
    )
    # regenieCond params
    parser.add_argument(
        "--summary",
        dest="summary",
        required=False,
        type=str,
        help="regenie format summary file path",
    )
    parser.add_argument(
        "--outputFolder",
        required=False,
        type=str,
        help="output folder path",
        default="regenie_cond_output",
    )
    parser.add_argument(
        "--disable-exclude-mode",
        dest="disable-exclude-mode",
        required=False,
        action="store_false",
        help="disable exclude mode",
    )
    parser.add_argument(
        "--max-condsnp",
        dest="max-condsnp",
        required=False,
        type=int,
        help="max conditional snp, default -1, will not max condsnp limit",
        default=-1,
    )
    parser.add_argument(
        "--condsnp-path",
        dest="condsnp-path",
        required=False,
        type=str,
        help="conditional snp list file path",
    )
    parser.add_argument(
        "--condsnp-list",
        dest="condsnp-list",
        required=False,
        nargs="+",
        help="conditional snp list",
    )

    parser.add_argument(
        "--defaultLOG10P", required=False, type=int, help="default log10p", default=6
    )
    parser.add_argument(
        "--defaultFREQ", required=False, type=float, help="default freq", default=1e-2
    )
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    args_dict = vars(args)

    regenieCond = RegenieConditionalAnalysis(args_dict)
    regenieCond.perform_conditional_analysis()
    # regenie = Regenie(**args_dict)
    # regenie()


if __name__ == "__main__":
    main()
