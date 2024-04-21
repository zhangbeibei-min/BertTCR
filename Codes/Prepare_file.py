#!/usr/bin/env python
# encoding: utf-8

import argparse
import os


def create_parser():
    parser = argparse.ArgumentParser(
        description="Script to process raw files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input_dir",
        dest="input_dir",
        type=str,
        help="The directory containing input raw files in .txt format.",
        default='/data/zhangm/BertTCR/RawData/Health',
    )
    parser.add_argument(
        "--reference",
        dest="reference",
        type=str,
        help="The reference dataset in .tsv format.",
        default="",
    )
    parser.add_argument(
        "--tcr_num",
        dest="tcr_num",
        type=int,
        help="The number of TCRs extracted from each file after filteration process.",
        default=100,
    )
    parser.add_argument(
        "--info_index",
        dest="info_index",
        type=list,
        help="The index list of the used information in each file.",
        default=[0, -1],
    )
    parser.add_argument(
        "--output_dir",
        dest="output_dir",
        type=str,
        help="Output directory for converted files.",
        default='/data/zhangm/BertTCR/RawData/Healthtsv',
    )
    args = parser.parse_args()
    return args


def txt_to_tsv(txt_file, output_dir):
    # 将txt文件转换为tsv文件
    with open(txt_file, 'r',encoding='gbk') as txt:
        lines = txt.readlines()
    tsv_file = os.path.splitext(os.path.basename(txt_file))[0] + '.tsv'
    tsv_path = os.path.join(output_dir, tsv_file)
    with open(tsv_path, 'w') as tsv:
        for line in lines:
            tsv.write(line.replace(' ', '\t'))

    return tsv_path


def read_tsv(filename, inf_ind, skip_1st=False, file_encoding="utf8"):
    # Return n * m matrix "final_inf" (n is the num of lines, m is the length of list "inf_ind").
    extract_inf = []
    with open(filename, "r", encoding=file_encoding) as tsv_f:
        if skip_1st:
            tsv_f.readline()
        line = tsv_f.readline()
        while line:
            line_list = line.strip().split("\t")
            temp_inf = []
            for ind in inf_ind:
                temp_inf.append(line_list[ind])
            extract_inf.append(temp_inf)
            line = tsv_f.readline()
    return extract_inf


def filter_sequence(raw, reference):
    # Filtering low-quality sequences and the ones in the reference dataset.
    aa_list = ["A", "R", "N", "D", "C", "E", "Q", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]
    result = []
    save_flag = 1
    for seq in raw:
        if len(seq[0]) > 24 or len(seq[0]) < 10:
            save_flag = 0
        for aa in seq[0]:
            if aa not in aa_list:
                save_flag = 0
        if seq[0][0].upper() != "C" or seq[0][-1].upper() != "F":
            save_flag = 0
        if [seq[0]] in reference:
            save_flag = 0
        if save_flag == 1:
            result.append(seq)
        else:
            save_flag = 1
    return result


if __name__ == "__main__":
    # Parse arguments.
    args = create_parser()

    # Create output directory.
    os.makedirs(args.output_dir, exist_ok=True)

    # Convert txt files to tsv files and filter TCRs in each file.
    for txt_file in os.listdir(args.input_dir):
        if txt_file.endswith('.txt'):
            txt_path = os.path.join(args.input_dir, txt_file)
            tsv_path = txt_to_tsv(txt_path, args.output_dir)
            raw_file = read_tsv(tsv_path, args.info_index, True)
            # Read the reference dataset.
            if args.reference != "":
                reference_file = read_tsv(args.reference, [0])
            else:
                reference_file = []
            # Extract TCRs.
            processed_file = filter_sequence(raw_file, reference_file)
            output_file = sorted(processed_file, key=lambda x: float(x[1]), reverse=True)[:args.tcr_num]
            # Save output.
            output_file_path = os.path.join(args.output_dir, os.path.basename(tsv_path))
            with open(output_file_path, "w", encoding="utf8") as output_f:
                output_f.write("TCR\tAbundance\n")
                for tcr in output_file:
                    output_f.write("{0}\t{1}\n".format(tcr[0], tcr[1]))
            print(f"The processed file '{output_file_path}' has been saved.")

