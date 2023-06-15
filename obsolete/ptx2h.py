#!/usr/bin/env python3
"""
:file:      ptx2h.py
:brief:     reads ptx (in fact, any text) file and puts its contents into const c-string in a generated header file
:date:      25.06.2021
:author:    Pavel Paramonov
            imec VisionLab
            University of Antwerp
            pavel.paramonov@uantwerpen.be
"""
import sys, getopt
from datetime import datetime

script_name = sys.argv[0]

def print_help_message():
    print(f"Usage: {script_name} -i <ptx_file> -o <header_file")

def parse_args(argv):
    inputfile = ''
    outputfile = ''
    try:
        opts, args = getopt.getopt(argv,"hi:o:",["ptx=","header="])
    except getopt.GetoptError:
        print_help_message()
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print_help_message()
            sys.exit()
        elif opt in ("-i", "--ptx"):
            inputfile = arg
        elif opt in ("-o", "--header"):
            outputfile = arg
    return inputfile, outputfile

if __name__ == "__main__":
    ptx_file_name, header_file_name = parse_args(sys.argv[1:])

    ptx_lines = []
    try:
        with open(ptx_file_name, 'r') as ptx_file:
            ptx_lines = ptx_file.readlines()
    except IOError:
        print(f"Error: could not read from file {ptx_file_name}")
        sys.exit(1)

    if len(ptx_lines) < 1:
        print(f"Error: input file {ptx_file_name} is empty")
        sys.exit(3)

    try:
        with open(header_file_name, 'w') as header_file:
            current_date = datetime.today()
            comment_line=f"// This file was generated with {script_name}, time stamp [{current_date}]\n"
            header_file.write(comment_line)
            header_file.write("\n")
            include_line="#include <string>"
            header_file.write(include_line)
            header_file.write("\n")
            ptx_string_variable_postfix = ptx_file_name.split('/')[-1].split('.')[0]
            ptx_string_variable = f"std::string ptx_string_{ptx_string_variable_postfix} = "
            for i in range(len(ptx_lines)-1):
                ptx_string_variable += ''.join(["std::string", "(\"", ptx_lines[i][:-1].replace("\"", "\\\""), '\\n', "\")", " + "])
            ptx_string_variable += ''.join(["std::string", "(\"", ptx_lines[-1][:-1].replace("\"", "\\\""), "\")"])
            ptx_string_variable += ";"
            header_file.write(ptx_string_variable)
    except IOError:
        print(f"Error: could not write to file {header_file_name}")