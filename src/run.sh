# 2020.09.18
# @yifan
#
# for fast run the encode and decode function

echo ""
echo 2020.09.07 @yifan
echo Run JPEG encode and decode function

#############################################
# raw code or modified code
#mode="jpeg-6b-raw"
mode="jpeg-6b-our"
echo $mode

# output folder name
#res_folder='jpeg_raw'


# image count, named start from 0.bmp
img_count=24

# dataset name
dataset='Kodak'
#dataset='DIV2K'
#dataset="NormalizedBrodatz"

# quality factor
N_start=90
res_folder='@N='$N_start
N_end=$N_start+2
N_step=10

# path
root="../"
input=$root"data/$dataset/"
output=$root"result/$dataset/$res_folder/"


#############################################

echo ""
echo "compiling.."
echo ""
cd $mode
./configure
make
cd ..
echo ""

if [ ! -d $root"result/"$dataset ] 
then
    mkdir $root"result/"$dataset
fi
if [ ! -d $input ] 
then
    mkdir $input
fi
if [ ! -d $output ] 
then
    mkdir $output
fi
if [ ! -d $output ] 
then
    mkdir $output
fi
if [ ! -d "$output"/encode ] 
then
    mkdir "$output"/encode
fi
if [ ! -d "$output"/decode ] 
then
    mkdir "$output"/decode
fi

for ((N=N_start; N <=N_end; N=N+N_step))
do
    for (( i=0; i < $img_count; i+=1 ))
    do
        echo Finish image $i @Qf=$N
        ./"$mode"/cjpeg -dct float -quality "$N" -outfile "$output"encode/"$i"_"$N".jpg  "$input$i".bmp
        ./"$mode"/djpeg -dct float -bmp -outfile "$output"decode/"$i"_"$N".bmp  "$output"encode/"$i"_"$N".jpg
    done
done