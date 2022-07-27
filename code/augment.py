# Easy data augmentation techniques for text classification
# sorce: Jason Wei and Kai Zou
# Added csv input 27th July 2022 KavyaD

from eda import *
import pandas as pd
from pathlib import Path
import nltk
nltk.download('omw-1.4')
#arguments to be parsed from command line
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--input", required=False, type=str, help="input file of unaugmented data")
ap.add_argument("--output", required=False, type=str, help="output file of unaugmented data")
ap.add_argument("--num_aug", required=False, type=int, help="number of augmented sentences per original sentence")
ap.add_argument("--alpha_sr", required=False, type=float, help="percent of words in each sentence to be replaced by synonyms")
ap.add_argument("--alpha_ri", required=False, type=float, help="percent of words in each sentence to be inserted")
ap.add_argument("--alpha_rs", required=False, type=float, help="percent of words in each sentence to be swapped")
ap.add_argument("--alpha_rd", required=False, type=float, help="percent of words in each sentence to be deleted")
ap.add_argument("--col_to_augment", required=True, type=str, help="column to augment")
args = ap.parse_args()

#the output file
output = None
if args.output:
    output = args.output
else:
    from os.path import dirname, basename, join
    output = join(dirname(args.input), 'eda_' + basename(args.input))


file_name = 'DataToAugment.csv'
try:
    file_name = str(args.input)
    print('Input file : ', file_name)
except:
    file_name = "data/"+file_name
    print('Invalid path, selecting default value: ', file_name)

path = Path(__file__).parent / file_name
dataset = pd.read_csv(path)

#number of augmented sentences to generate per original sentence
num_aug = 9 #default
if args.num_aug:
    num_aug = args.num_aug

#how much to replace each word by synonyms
alpha_sr = 0.1#default
if args.alpha_sr is not None:
    alpha_sr = args.alpha_sr

#how much to insert new words that are synonyms
alpha_ri = 0.1#default
if args.alpha_ri is not None:
    alpha_ri = args.alpha_ri

#how much to swap words
alpha_rs = 0.1#default
if args.alpha_rs is not None:
    alpha_rs = args.alpha_rs

#how much to delete words
alpha_rd = 0.1#default
if args.alpha_rd is not None:
    alpha_rd = args.alpha_rd

if alpha_sr == alpha_ri == alpha_rs == alpha_rd == 0:
     ap.error('At least one alpha should be greater than zero')


data = []

#generate more data with standard augmentation
def gen_eda_csv(row_,train_orig, output_file, alpha_sr, alpha_ri, alpha_rs, alpha_rd, num_aug=9):
    print(row_[args.col_to_augment])
    aug_sentences = eda(row_[args.col_to_augment], alpha_sr=alpha_sr, alpha_ri=alpha_ri, alpha_rs=alpha_rs, p_rd=alpha_rd, num_aug=num_aug)
    print("generated augmented sentences with eda for " + train_orig + " to " + str(output_file) + " with num_aug=" + str(num_aug))
    global data
    for aug_sentence in aug_sentences:
        row_[args.col_to_augment] = aug_sentence
        data.append(row_.values)
    return True

#main function
if __name__ == "__main__":
    
    #generate augmented sentences and output into a new file
    eda_data_check = (dataset.apply(lambda x: gen_eda_csv(x,args.input, output, alpha_sr=alpha_sr, alpha_ri=alpha_ri, alpha_rs=alpha_rs, alpha_rd=alpha_rd, num_aug=num_aug), axis=1))
    data_columns = dataset.columns
    data_aug = pd.DataFrame(data,columns=data_columns)
    data_aug.to_csv(output)
    print("Done.")