"""Script to visualise the captions that Oscar spits out."""
import glob
import os

import click
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image


@click.command()
@click.argument("image_dir")
@click.argument("oscar_results_tsv")
def main(image_dir, oscar_results_tsv):
    """Show images with captions."""
    files = glob.glob(os.path.join(image_dir, "*"))

    df = pd.read_csv(oscar_results_tsv, sep="\t", header=None)

    for file in files:
        im_id = file.split("/")[-1].split(".")[0]
        im = Image.open(file)
        plt.imshow(im)
        caption = eval(df[df[0] == im_id].iloc[0][1])[0]["caption"]
        plt.title(caption)
        plt.axis("off")
        plt.savefig(".".join(file.split(".")[:-1]) + "_result.png")
        plt.clf()
        plt.close()


if __name__ == "__main__":
    main()
