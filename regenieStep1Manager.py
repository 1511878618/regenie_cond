#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Description:       :
@Date     :2024/02/20 14:06:15
@Author      :Tingfeng Xu
@version      :1.0
'''

from pathlib import Path
import sys
try:
    import pandas as pd 
except ImportError:
    import subprocess
    import sys

    print("缺少pandas模块,开始安装...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "pandas"], check=True)
        print("successfully installed pandas")
        import pandas as pd 
    except subprocess.CalledProcessError:
        print("unable to install pandas, please install it manually")
        sys.exit(1)

import argparse
import shutil
import textwrap
import warnings
try:
    from rich.console import Console
    from rich.table import Table
except:
    print("缺少rich模块,开始安装...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "rich"], check=True)
        print("successfully installed rich")
        from rich.console import Console
        from rich.table import Table
    except subprocess.CalledProcessError:
        print("unable to install rich, please install it manually")
        sys.exit(1)

class DataFramePretty(object):
    def __init__(self, df: pd.DataFrame) -> None:
        self.data = df

    def show(self):
        table = Table()

        # self.data是原始数据
        # df 是用来显示的数据
        df = self.data.copy()
        for col in df.columns:
            df[col] = df[col].astype("str")
            table.add_column(col)

        for idx in range(len(df)):
            table.add_row(*df.iloc[idx].tolist())

        console = Console()
        console.print(table)


class RegenieStep1:

    def __init__(self, step1_list_dir=None, step1_df=None) -> None:
        """
        Assume the loco file is in the same folder with the pred.list file or  path in the pred.list file is correct

        When init will auto check the loco file path record in the pred.list file, if file not exists, will warning and  try to find the matched loco file in the same folder, and if still not found, will raise error

        So if founded, then run fix will update


        """
        if step1_df is not None:
            self.step1_df = step1_df
        elif step1_list_dir is not None:
            self.step1_list_dir = step1_list_dir
            self.step1_df = pd.read_csv(
                self.step1_list_dir, sep="\s+", header=None, names=["phenotype", "path"]
            )
        else:
            raise ValueError("step1_list_dir or step1_df should be provided")

        self._build()  # build the step1 manager

    def _build(self):

        # check the loco file path record in the pred.list file

        new_step1_df = self.step1_df.copy()
        for idx, row in self.step1_df.iterrows():
            # check the path exists
            if not Path(row['path']).exists():
                sys.stdout.write(
                    f"{row['phenotype']} loco file {row['path']} not exists, will try to find it in local loco files to match the phenotype\n"
                )
                new_path = self.step1_list_dir.parent / f"{row['phenotype']}.loco"
                if new_path.exists():
                    new_step1_df.loc[idx, "local_path"] = new_path.resolve()
                    new_step1_df.loc[idx, "status"] = 1
                else:
                    sys.stderr.write(
                        f"Can not find the matched loco file for {row['phenotype']} \n"
                    )
                    new_step1_df.loc[idx, "status"] = 0
                    new_step1_df.loc[idx, "local_path"] = None
            else:
                new_step1_df.loc[idx, "local_path"] = Path(row["path"]).resolve()
                new_step1_df.loc[idx, "status"] = 1
        self.step1_df = new_step1_df

    def fix(self):
        """
        fix the path in the pred.list file; find the matched loco file in the step1.list same folder
        """
        # only keep the status == 1

        DataFramePretty(self.step1_df).show()
        status_ok = self.step1_df.query("status == 1").shape[0]
        status_not_ok = self.step1_df.query("status == 0").shape[0]
        sys.stdout.write(f"status ok: {status_ok}, status not ok: {status_not_ok}\n")
        sys.stdout.write("Will fix the path in the pred.list file\n")
        self.step1_df = self.step1_df.query("status == 1")
        # save
        # self.step1_df[["phenotype", "local_path"]].to_csv(
        #     self.step1_list_dir, sep=" ", index=False, header=False
        # )
        # reload
        self._build()

    def cp(self, tgt_dir):
        # copy all loco files to the new folder
        tgt_dir = Path(tgt_dir)
        if not tgt_dir.exists():
            tgt_dir.mkdir(parents=True, exist_ok=True)
        new_step1_df = self.step1_df.copy()
        for idx, row in self.step1_df.iterrows():
            # new path
            new_path = tgt_dir / row["phenotype"]

            if new_path.exists():
                raise ValueError(f"{new_path} already exists")
            else:
                shutil.copyfile(row["local_path"], new_path)
                new_step1_df.loc[idx, "local_path"] = new_path
        print("All loco files copied to the new folder")

    def save(self, new_step1_dir, cp=False):
        # save the step1 file to new dir

        # check basic file
        new_step1_dir = Path(new_step1_dir)
        if new_step1_dir.exists():
            raise ValueError(f"{new_step1_dir} already exists")
        new_step1_dir.parent.mkdir(parents=True, exist_ok=True)

        # save the step1 file
        self.step1_df[["phenotype", "local_path"]].to_csv(
            new_step1_dir, sep=" ", index=False, header=False
        )

        # copy all loco files to the new folder
        if cp:
            parent_dir = new_step1_dir.parent
            self.cp(parent_dir)


def concatRegeineStep1(to_merge_list, force=False):
    """
    concat RegenieStep1 object

    """
    # TODO: merge the other list to the first one by only concat and keep local_path is ok
    assert all(
        [isinstance(x, RegenieStep1) for x in to_merge_list]
    ), "all to merge list should be RegenieStep1 object"

    # for each_step1 in to_merge_list:
    #     # check status col in the step1_df
    #     if "status" not in each_step1.step1_df.columns:
    #         sys.stderr.write(
    #             "status col not found in the step1_df, please run fix first\n"
    #         )
    #         exit(1)

    #     if each_step1.step1_df.query("status == 0").shape[0] > 0:
    #         sys.stderr.write(
    #             "status not ok found in the step1_df, please run fix first\n"
    #         )
    #         exit(1)

    for each_step1 in to_merge_list:
        each_step1.fix()

    # merge the pred.list files
    merged_step1_df = pd.concat([each_step1.step1_df for each_step1 in to_merge_list])

    # check any duplicated phenotype
    duplicated = merged_step1_df.duplicated(subset=["phenotype"])
    is_duplicated = duplicated.any()

    if is_duplicated:
        sys.stdout.write(
            f"Found {duplicated.sum()} duplicated loco files with duplicated name\n"
        )
        duplicated_df = merged_step1_df[duplicated]
        DataFramePretty(duplicated_df).show()

        if not force:
            sys.stderr.write(
                "Duplicated phenotype found, can not merge them, please check the pred.list files in each step1 folder or use --force option to force merge them\n"
            )
            exit(1)
        else:
            # rename the duplicated phenotype with suffix
            sys.stdout.write("Force merge them with --force passed\n")
            merged_step1_df["old_phenotype"] = merged_step1_df["phenotype"]
            merged_step1_df["phenotype"] = (
                merged_step1_df["path"].apply(lambda x: Path(x).parent.name)
                + "_"
                + merged_step1_df["phenotype"]
            )
            # DataFramePretty(merged_step1_df.sort_values(by="old_phenotype")).show()

    # return new step1 object
    merged_RegeineStep1 = RegenieStep1(step1_df=merged_step1_df)
    # fix
    merged_RegeineStep1.fix()
    return merged_RegeineStep1


def getParser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(
            """
            %prog is ...
            @Author: xutingfeng@big.ac.cn
            Version: 1.0


            # list step1 
            %prog -i step1_1.list step1_2.list -l

            # fix the path in pred.list file; find the matched loco file in the step1.list same folder 
            %prog -i step1_1.list step1_2.list --fix 

            # merge the pred.list files in different step1 folders; Note: if any of them are same name, will raise errors, use --force to force merge them with suffix; or save each at separate folder
            %prog -i step1_1.list step1_2.list -m -o new_step1.list

            # if you want to merge and cp all loco file to new step1.list folder 
            %prog -i step1_1.list step1_2.list -m -o new_step1.list --cp
            
            # same --update option to update the other to the first one
            %prog -i step1_1.list step1_2.list --update 

            # --cp will cp all loco files to the new folder
            %prog -i step1_1.list step1_2.list --update --cp
                
            """
        ),
    )

    parser.add_argument('-i', '--input', default=['step1'], help='input folder of step1, -i step1_1 step1_2', nargs='+')
    parser.add_argument('-l', '--list', help='list all phenotypes', action='store_true')
    parser.add_argument(
        "--fix",
        help="fix the path in pred.list file; find the matched loco file in the step1.list same folder",
        action="store_true",
    )
    parser.add_argument('-m', '--merge', help='merge all pred.list files', action='store_true')
    parser.add_argument("-o", "--output", type=str, help="output folder", default=None)
    parser.add_argument('--force', help='force merge the pred.list files', action='store_true')
    parser.add_argument('--update', help='update the pred.list files, only use with one input and update step1 file structure with new dir', action='store_true')
    parser.add_argument(
        "--cp", help="copy all loco files to the new folder", action="store_true"
    )

    return parser


if __name__ == "__main__":
    parser = getParser()
    args = parser.parse_args()
    # For Pretrain dict
    input_folders = args.input

    step1s = [RegenieStep1(step1_list_dir=stepFolder) for stepFolder in input_folders]

    for step1 in step1s:
        if args.fix:
            step1.fix()

            # update
            step1.save(step1.step1_list_dir, cp=False)

        if args.list:
            DataFramePretty(step1.step1_df).show()

    if args.merge:
        sys.stdout.write("Will merge all step1 folders\n")
        sys.stdout.write("-" * 20 + "\n")
        main_object = concatRegeineStep1(step1s, args.force)

        if args.list:
            DataFramePretty(main_object.step1_df).show()

    elif len(step1s) == 1 and args.output is not None:
        sys.stdout.write("Only one step1 folder found\n")
        main_object = step1s[0]

    if args.output is not None:
        main_object.save(args.output, cp=args.cp)
