#!/bin/bash
files=(
    "exame1/OUL0CZ5E/HDVUJOM3"
    "exame1/OUL0CZ5E/GRUPKMLN"
    "exame2/SXDWNPE2/NMKW5E5D"
    "exame2/SXDWNPE2/JXMQZVMI"
    "exame3/ES2GJ21P/0BPBD50Q"
    "exame3/ES2GJ21P/2OPQL5YQ"
    "exame4/VQ2KRLB2/AA1R42FR"
    "exame4/VQ2KRLB2/K55KYYBG"
    "exame5/FQMAC1DT/CRJN1X2T"
    "exame5/FQMAC1DT/D2KSTXMT"
    "exame6/RJY5FFRD/JJA10ZCU"
    "exame6/RJY5FFRD/LUTTOPM0"
    "exame7/FAMDN1BX/C0LGTVKT"
    "exame7/FAMDN1BX/ATFQ1VMT"
    "exame8/FACCYNVK/PW4YCRXA"
    "exame8/FACCYNVK/OX4JSRJA"
    "exame9/3XSGMZP0/J1VSOESL"
    "exame9/3XSGMZP0/KLBWYWNU"
    "exame10/KRK2JGZK/YMTSVLAD"
    "exame10/KRK2JGZK/5GUANL4D"
    "exame11/QM2OECZS/UFLR0JKB"
    "exame11/QM2OECZS/LE4LKLM2"
    "exame12/5OP45XQO/A3KB4TPB"
    "exame12/5OP45XQO/A345EPXO"
    "exame13/AKUN305M/MV4VSS1H"
    "exame13/AKUN305M/O0ZQCS5H"
    "exame14/IHF43LRS/AKPI2I5W"
    "exame14/IHF43LRS/GCCJIB34"
    "exame15/R0Q1RRHU/NAVFPTTP"
    "exame15/R0Q1RRHU/RJESORWI"
    "exame16/32VP12L3/LOSTUIJI"
    "exame16/32VP12L3/PVFJ4OYU"
    "exame17/L5LQFSIF/L1XNQ3BP"
    "exame17/L5LQFSIF/RCD5GD4X"
    "exame18/QURAI2ZN/DDUSUKIP"
    "exame18/QURAI2ZN/Q13X3X52"
    "exame19/L1VEY5ZJ/IDPXNPAM"
    "exame19/L1VEY5ZJ/ITOCNPCM"
    "exame20/VCEQWILJ/QIVCY5BL"
    "exame20/VCEQWILJ/VJUJY5DL"
)
python_command="python start_segmentation.py --save-mask --save-images --save-overlap"
input_folder="$1"
output_folder="$2"

if [ -z "$1" ] || [ -z "$2" ]; then
    printf "No input/output folder supplied"
    exit
fi

for file in ${files[*]}; do 
    command="$python_command $input_folder/$file $output_folder/$file";
    printf "Running $command\n";
    
    $command;
    printf "\n---------------\n";
done;