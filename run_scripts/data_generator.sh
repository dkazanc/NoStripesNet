#!/bin/bash
mkdir ../DATA_UNET_TOMO
mkdir ../DATA_UNET_TOMO/TESTDATA

mkdir ../DATA_UNET_TOMO/recon
rm -rf ../DATA_UNET_TOMO/recon/*
mkdir ../DATA_UNET_TOMO/gt
rm -rf ../DATA_UNET_TOMO/gt/*

mkdir ../DATA_UNET_TOMO/TESTDATA/recon
rm -rf ../DATA_UNET_TOMO/TESTDATA/recon/*
mkdir ../DATA_UNET_TOMO/TESTDATA/gt
rm -rf ../DATA_UNET_TOMO/TESTDATA/gt/*
########### CHANGE PARAMETERS HERE ###########
output_path_to_reconstructions="../DATA_UNET_TOMO/recon/"
output_path_to_masks="../DATA_UNET_TOMO/gt/"
number_of_datasets=15 # set how many reconstructed datasets you would like to generate
volume_recon_size=128 # this will create a 3D volumes of volume_recon_size
total_projections=160 # the total number of projections to generate
reconstruction_algorithm="FBP" # choose between "FBP" and "ITERAT"
#############################################

echo "The training data generation has started"
python synth_data_gen/synth_data_generator.py -i $output_path_to_reconstructions -m  $output_path_to_masks -n $number_of_datasets -s $volume_recon_size -a $total_projections -alg $reconstruction_algorithm
echo "The data has been generated"

echo "The test data generation has started"
output_path_to_reconstructions="../DATA_UNET_TOMO/TESTDATA/recon/"
output_path_to_masks="../DATA_UNET_TOMO/TESTDATA/gt/"
number_of_datasets=1 # set how many TEST datasets you would like to generate
python synth_data_gen/synth_data_generator.py -i $output_path_to_reconstructions -m  $output_path_to_masks -n $number_of_datasets -s $volume_recon_size -a $total_projections -alg $reconstruction_algorithm
echo "The data has been generated"
