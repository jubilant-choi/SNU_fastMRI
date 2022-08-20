python ./train.py -u JB -x UnetPP_CV1 -n UnetPP -b 8 -e 25 -l 1e-4 -s P --input-key reconstruction --CV 1 --target-key image_label_cmask --load UnetPP_CV1_best.pt

for i in {2..5}
do
python ./train.py -u JB -x UnetPP_CV${i} -n UnetPP -b 8 -e 30 -l 1e-4 -s P --input-key reconstruction --CV $i --target-key image_label_cmask # --load UnetPP_CV${i}_best.pt
exit
done
