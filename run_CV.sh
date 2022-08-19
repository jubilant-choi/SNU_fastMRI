python ./train.py -u JB -x Unet_CVtest -n Unet -b 12 -e 5 -l 1e-4 -s P --input-key reconstruction --CV 1 --load Unet_CVtest_best.pt
for i in {2..5}
do
python ./train.py -u JB -x Unet_CV${i} -n Unet -b 12 -e 5 -l 1e-4 -s P --input-key reconstruction --CV $i --load Unet_CV${i}_best.pt
done
