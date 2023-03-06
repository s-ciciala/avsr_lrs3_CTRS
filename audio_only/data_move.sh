#!/bin/bash

# Define the paths to the dataset directory and the destination directory
DATASET_DIR=/group/corpora/public/lipreading/LRS3
DEST_DIR=../../lrs3

mkdir -p $DEST_DIR


# Loop through each file in the dataset directory and create a symbolic link
for file_path in $(find $DATASET_DIR -type f); do
  # Determine the relative path to the file from the dataset directory
  rel_path=$(realpath --relative-to=$DATASET_DIR $file_path)

  # Create the destination directory for the file, if necessary
  dest_subdir=$(dirname $rel_path)
  mkdir -p "$DEST_DIR/$dest_subdir"

  # Create the symbolic link
  ln -s $file_path "$DEST_DIR/$rel_path"
done
