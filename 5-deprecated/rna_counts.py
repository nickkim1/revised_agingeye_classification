import argparse as ap
import gffutils
import pyranges as pr
import pandas as pd

# MAKE API CALL USING REQUESTS: https://requests.readthedocs.io/en/latest/
    # - load in the data I downloaded from flybase 
    # - make requests using the second column identifier 
        # - make sure i don't get banned 
        # - extract chromosomal coord information 
# worst comes to worst web scrape using beautiful soup

# merge peaks for rnaseq data
parser = ap.ArgumentParser(prog='This program processes in RNAseq data into coverage format, binning into 600 bp bins')
parser.add_argument('-gtf_filepath', type=str, help='Input filepath of GTF file to process', nargs=1)
parser.add_argument('-counts_filepath', type=str, help='Input filepath of counts file to process', nargs=1)
parser.add_argument('-sequence_length', type=int, help="Input the length of the sequence you want for each gene")
args = vars(parser.parse_args())
gtf_filepath = args['gtf_filepath'][0]
counts_filepath = args['counts_filepath'][0]
sequence_length = args['sequence_length']

# first, load in the gtf file 
gr = pr.read_gtf(gtf_filepath)

# convert to pandas dataframe 
df = gr.df

# do some feature exploration 
print("=====================================================================================================================")
print("These are the first two entries of the GTF file, in dataframe format")
print(df.head(2))

# filter dataframe by "gene"
print("=====================================================================================================================")
print("These are now the first two entries of the GTF file, in dataframe format, after filtering for only genes")
df = df[df['Feature'] == 'gene']
print(df.head(5))

# now retrieve the TSS coordinates. 
    # * start/end coords are given for DNA strand 
    # * 1 - if gene is on + (coding) strand, then it is oriented 5' -> 3' and so start position on xsome aligns w/ TSS of gene (which is transcribed 5'->3')
    # * 2 - if gene is on - (template) strand, then it is oriented 3' -> 5' so end position on chromosome is on the 5' end of gene and is actually where transcription starts
    # thus, + has TSS = start. - has TSS = end. 

tss_sites = pd.Series([df.iloc[i, 3] if df.iloc[i, 6] == "-" else df.iloc[i, 4] for i in range(df.shape[0])])
print("=====================================================================================================================")
print("This is a sample of the appropriate TSS sites in the dataframe")
print(tss_sites.head(5))

# setup new start and end coordinates, set this to whatever you want 
tss_relative_starts = tss_sites - (sequence_length // 2)
tss_relative_ends = tss_sites + (sequence_length // 2)

# now create a new dataframe with the chromosome, gene id, start and stop 
tss_df = pd.DataFrame({
    "Chromosome": df["Chromosome"].values,
    "gene_id": df["gene_id"].values, 
    "Start": tss_relative_starts, 
    "Stop": tss_relative_ends
})

print("=====================================================================================================================")
print("This is the original df post-TSS concatenation")
print(tss_df.head(5)) 
