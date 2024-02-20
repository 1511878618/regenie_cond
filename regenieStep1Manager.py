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
    def __init__(self, stepFolder=None, pred_df=None) -> None:
        self.dir = stepFolder
        if pred_df is not None:
            self.pred_df = pred_df
        else:
            self._build() # build the step1 manager

    def _build(self):
        self.locos = list(Path(self.dir).glob("*.loco"))
        self.pred_list = list(Path(self.dir).glob("*pred.list"))
        if len(self.pred_list) == 0:
            raise ValueError(f"No pred list file found in {self.dir}")
        elif len(self.pred_list) > 1:
            # raise ValueError(f"More than one pred list file found in {self.dir}, please use --merge option to merge them")
            sys.stdout.write(f'More than one pred list file found in {self.dir}, will merge them\n')

        pred_df = pd.concat([pd.read_csv(pred, sep="\s+", header=None, names=['phenotype', 'path']) for pred in self.pred_list])

        for idx, row in pred_df.iterrows():
            if not Path(row['path']).exists():
                sys.stdout.write(f"{row['path']} not exists, will try to find it in local loco files to match the phenotype\n")
                wrong_path = row['path']
                wrong_base_name = Path(wrong_path).name
                matched_loco = [str(loco.resolve()) for loco in self.locos if wrong_base_name in loco.name]
                if len(matched_loco) == 1:
                    pred_df.loc[idx, 'path'] = matched_loco[0]
                    pred_df.loc[idx, 'old_path'] = wrong_path
                else:
                    raise ValueError(f"Can not find the matched loco file for {wrong_path} with {matched_loco}")
        

        self.pred_df = pred_df

    def move(self, outDir):
        """
        may copy is more correct to describe the opreate
        """
        outDir = Path(outDir)
        outDir.mkdir(parents=True, exist_ok=True)

        # mv 
        new_pred_df_list = []
        for idx, row in self.pred_df.iterrows():
            pheno = row['phenotype']
            path = row['path']
            new_path = outDir / f"{pheno}.loco"
            if new_path.exists():
                raise ValueError(f"{new_path} already exists")
            else:
                # Path(path).rename(new_path)
                shutil.copyfile(path, new_path)
                new_pred_df_list.append({'phenotype': pheno, 'path': new_path.resolve()})
        # save new pred_df  
        self.pred_df = pd.DataFrame(new_pred_df_list)
        self.dir = outDir
        new_pred_path = Path(self.dir) / "pred.list"
        self.pred_df[['phenotype', 'path']].to_csv(new_pred_path, sep=" ", index=False, header=False)


    def save(self):
        new_pred_path = Path(self.dir) / "pred.list"
        for old_pred in self.pred_list:
            old_pred.rename(str(old_pred) + '.old')
        self.pred_list = [new_pred_path]
        self.pred_df[['phenotype', 'path']].to_csv(new_pred_path, sep=" ", index=False, header=False)




def getParser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(
            """
            %prog is ...
            @Author: xutingfeng@big.ac.cn
            Version: 1.0
            supported usage:
            1. list the step1 file structure; note you can use -i step1_1 step1_2 to list multiple step1 file structure; if the path is not correct, there will use path and old path to mark them (if these file indeed exists)
                %prog -i step1_1 -l
            2. If you found the path is not correct in xx_pred.list file and -l can help you match the real path, then use --update to update the pred.list file 
                %prog -i step1 --udpate
                
                also you can use -o to specify the output folder instead of update at the original folder

                %prog -i step1 -o new_step1

            3. This can work with multiple step1 folders, if you only want to update them each other
                % prog -i step1_1 step1_2 -l # show 
                or 
                % prog -i step1_1 step1_2 --update # update 
            4. You can merge the pred.list files in different step1 folders, but first use -l to list final merged 
                %prog -i step1_1 step1_2 -m -l 
            5. If there are some duplicates in the merged pred.list file, you can use --force to force merge them, but with --force, the old phenotype will be added to the new phenotype as suffix; -l will show the final merged pred.list file
                %prog -i step1_1 step1_2 -m --force -l
            6. without -l option, the final merged pred.list file will be moved to new_step1 folder (can be changed by -o option)
                %prog -i step1_1 step1_2 -m --force -o new_step1

                
            """
        ),
    )
    parser.add_argument('-i', '--input', default=['step1'], help='input folder of step1, -i step1_1 step1_2', nargs='+')
    parser.add_argument('-l', '--list', help='list all phenotypes', action='store_true')
    parser.add_argument('-m', '--merge', help='merge all pred.list files', action='store_true')
    parser.add_argument('-o', '--output', type=str, help='output folder', default='new_step1')
    parser.add_argument('--force', help='force merge the pred.list files', action='store_true')
    parser.add_argument('--update', help='update the pred.list files, only use with one input and update step1 file structure with new dir', action='store_true')


    return parser


if __name__ == "__main__":
    parser = getParser()
    args = parser.parse_args()
    # For Pretrain dict
    input_folders = args.input
    step1s = [RegenieStep1(stepFolder) for stepFolder in input_folders]


    if args.merge and len(step1s) > 1:
        sys.stdout.write("Merging files\n")
        merged_pred_df = pd.concat([step1.pred_df for step1 in step1s])

        is_duplicated = merged_pred_df.duplicated(subset=['phenotype']).any()
        if is_duplicated:
            sys.stdout.write("Duplicated phenotype found, can not merge them, please check the pred.list files in each step1 folder or use --force option to force merge them\n")

        if not args.force:
            DataFramePretty(merged_pred_df[merged_pred_df.duplicated(subset=['phenotype'], keep=False)].sort_values('phenotype')).show()
            sys.stdout.write("Use --force option to force merge them\n")
            exit(1)
        else:
            sys.stdout.write("Force merge them with --force passed\n")
            merged_pred_df['old_phenotype'] = merged_pred_df['phenotype']
            merged_pred_df['phenotype'] = merged_pred_df['path'].apply(lambda x: Path(x).parent.name) + "_" + merged_pred_df['phenotype']
            DataFramePretty(merged_pred_df.sort_values(by='old_phenotype')).show()

        merged_RegeineStep1 = RegenieStep1(pred_df=merged_pred_df)
    elif args.merge and len(step1s) == 1:
        warnings.warn("Only one step1 folder found, no need to merge")

    if len(step1s) == 1 and not args.merge:
        merged_RegeineStep1 = step1s[0]

    if args.list and not args.merge:
        sys.stdout.write('Listing all step1 files\n')
        for step1manager in step1s:
            sys.stdout.write(f"Step1 folder: {step1manager.dir}\n")
            DataFramePretty(step1manager.pred_df).show()
        exit(0)
    elif args.list and args.merge:
        DataFramePretty(merged_RegeineStep1.pred_df).show()
        exit(0)

    if not args.update:
        sys.stdout.write(f"Moving files to {args.output}\n")    
        merged_RegeineStep1.move(args.output)
    elif args.update:
        for step1 in step1s:
            step1.save()
            sys.stdout.write(f"upadte the pred.list file in {step1.dir}\n")





