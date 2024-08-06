import pandas as pd

def reorder_dataframe_rows(input, order_list, output):

    df = pd.read_csv(input)
    df = df.set_index(df.columns[0])

    # Reindex the DataFrame using the order_list
    reordered_df = df.loc[order_list]
    reordered_df = reordered_df.reset_index()
    
    reordered_df.to_csv(output, index=False)



# Order based on 4 categories
# Human, sequence classification
category_1 = ["GM12878", "HUVEC", "Hela-S3", "NHEK",  "nontata_promoter", 'prom_core_notata', 'prom_core_tata', 'prom_core_all','prom_300_notata', 'prom_300_tata', 'prom_300_all',
              "coding", "donors", "acceptors", "binary", "enhancer_cohn", "enhancer_ensembl", 'tf_0', 'tf_1', 'tf_2', 'tf_3', 'tf_4', "ocr",  "DNase_I"]
# Multispecies, sequence classification
category_2 = ["B_amyloliquefaciens", "R_capsulatus", "Arabidopsis_NonTATA", "Arabidopsis_TATA", "human_worm", 'mouse_0', 'mouse_1', 'mouse_2', 'mouse_3', 'mouse_4']
# Human, epigenetics modification
category_3 = ["5mC", "6mC"]
# Multispecies, epigenetics modification
category_4 = ["4mC_A.thaliana", "4mC_C.elegans", "4mC_D.melanogaster", "4mC_E.coli", "4mC_G.pickeringii", "4mC_G.subterraneus",
              'H3K79me3', 'H3K4me1', 'H4', 'H4ac', 'H3K4me2', 'H3K4me3', 'H3K14ac', 'H3K36me3', 'H3K9ac']
# Multi-class classification problems
multi = ["all", "multi", "reconstructed", "covid", "regulatory"]
order_list = category_1 + category_2 + category_3 + category_4 + multi


reorder_dataframe_rows("final_dnabert2.csv", order_list, "dnabert2_ordered.csv")
reorder_dataframe_rows("final_dnabert2_meanpool.csv", order_list, "dnabert2_meanpool_ordered.csv")
reorder_dataframe_rows("final_ntv2.csv", order_list, "ntv2_ordered.csv")
reorder_dataframe_rows("final_ntv2_meanpool.csv", order_list, "ntv2_meanpool_ordered.csv")
reorder_dataframe_rows("final_hyena.csv", order_list, "hyena_ordered.csv")
reorder_dataframe_rows("final_hyena_meanpool.csv", order_list, "hyena_meanpool_ordered.csv")



# 1st category: Human, sequence classification
# AUC and summary token pooling
db = pd.read_csv("dnabert2_ordered.csv", index_col=0)
db = db.rename(columns={"AUC": "DNABERT-2"})
db = (db.loc[category_1])["DNABERT-2"]

nt = pd.read_csv("ntv2_ordered.csv", index_col=0)
nt = nt.rename(columns={"AUC": "NT-v2"})
nt = (nt.loc[category_1])["NT-v2"]

hyena = pd.read_csv("hyena_ordered.csv", index_col=0)
hyena = hyena.rename(columns={"AUC": "HyenaDNA"})
hyena = (hyena.loc[category_1])["HyenaDNA"]

result = pd.concat([db, nt, hyena], axis=1)
result = result.round(3)
result.to_csv("./results_1/summary_token.csv")

# AUC and mean pooling
db = pd.read_csv("dnabert2_meanpool_ordered.csv", index_col=0)
db = db.rename(columns={"AUC": "DNABERT-2"})
db = (db.loc[category_1])["DNABERT-2"]

nt = pd.read_csv("ntv2_meanpool_ordered.csv", index_col=0)
nt = nt.rename(columns={"AUC": "NT-v2"})
nt = (nt.loc[category_1])["NT-v2"]

hyena = pd.read_csv("hyena_meanpool_ordered.csv", index_col=0)
hyena = hyena.rename(columns={"AUC": "HyenaDNA"})
hyena = (hyena.loc[category_1])["HyenaDNA"]

result = pd.concat([db, nt, hyena], axis=1)
result = result.round(3)
result.to_csv("./results_1/mean.csv")




# 2nd category: Other species, sequence classification
# AUC and summary token pooling
db = pd.read_csv("dnabert2_ordered.csv", index_col=0)
db = db.rename(columns={"AUC": "DNABERT-2"})
db = (db.loc[category_2])["DNABERT-2"]

nt = pd.read_csv("ntv2_ordered.csv", index_col=0)
nt = nt.rename(columns={"AUC": "NT-v2"})
nt = (nt.loc[category_2])["NT-v2"]

hyena = pd.read_csv("hyena_ordered.csv", index_col=0)
hyena = hyena.rename(columns={"AUC": "HyenaDNA"})
hyena = (hyena.loc[category_2])["HyenaDNA"]

result = pd.concat([db, nt, hyena], axis=1)
result = result.round(3)
result.to_csv("./results_2/summary_token.csv")

# AUC and mean pooling
db = pd.read_csv("dnabert2_meanpool_ordered.csv", index_col=0)
db = db.rename(columns={"AUC": "DNABERT-2"})
db = (db.loc[category_2])["DNABERT-2"]

nt = pd.read_csv("ntv2_meanpool_ordered.csv", index_col=0)
nt = nt.rename(columns={"AUC": "NT-v2"})
nt = (nt.loc[category_2])["NT-v2"]

hyena = pd.read_csv("hyena_meanpool_ordered.csv", index_col=0)
hyena = hyena.rename(columns={"AUC": "HyenaDNA"})
hyena = (hyena.loc[category_2])["HyenaDNA"]

result = pd.concat([db, nt, hyena], axis=1)
result = result.round(3)
result.to_csv("./results_2/mean.csv")



# 3rd category
# AUC and summary token pooling
db = pd.read_csv("dnabert2_ordered.csv", index_col=0)
db = db.rename(columns={"AUC": "DNABERT-2"})
db = (db.loc[category_3])["DNABERT-2"]

nt = pd.read_csv("ntv2_ordered.csv", index_col=0)
nt = nt.rename(columns={"AUC": "NT-v2"})
nt = (nt.loc[category_3])["NT-v2"]

hyena = pd.read_csv("hyena_ordered.csv", index_col=0)
hyena = hyena.rename(columns={"AUC": "HyenaDNA"})
hyena = (hyena.loc[category_3])["HyenaDNA"]

result = pd.concat([db, nt, hyena], axis=1)
result = result.round(3)
result.to_csv("./results_3/summary_token.csv")

# AUC and mean pooling
db = pd.read_csv("dnabert2_meanpool_ordered.csv", index_col=0)
db = db.rename(columns={"AUC": "DNABERT-2"})
db = (db.loc[category_3])["DNABERT-2"]

nt = pd.read_csv("ntv2_meanpool_ordered.csv", index_col=0)
nt = nt.rename(columns={"AUC": "NT-v2"})
nt = (nt.loc[category_3])["NT-v2"]

hyena = pd.read_csv("hyena_meanpool_ordered.csv", index_col=0)
hyena = hyena.rename(columns={"AUC": "HyenaDNA"})
hyena = (hyena.loc[category_3])["HyenaDNA"]

result = pd.concat([db, nt, hyena], axis=1)
result = result.round(3)
result.to_csv("./results_3/mean.csv")




# 4th category
# AUC and summary token pooling
db = pd.read_csv("dnabert2_ordered.csv", index_col=0)
db = db.rename(columns={"AUC": "DNABERT-2"})
db = (db.loc[category_4])["DNABERT-2"]

nt = pd.read_csv("ntv2_ordered.csv", index_col=0)
nt = nt.rename(columns={"AUC": "NT-v2"})
nt = (nt.loc[category_4])["NT-v2"]

hyena = pd.read_csv("hyena_ordered.csv", index_col=0)
hyena = hyena.rename(columns={"AUC": "HyenaDNA"})
hyena = (hyena.loc[category_4])["HyenaDNA"]

result = pd.concat([db, nt, hyena], axis=1)
result = result.round(3)
result.to_csv("./results_4/summary_token.csv")

# AUC and mean pooling
db = pd.read_csv("dnabert2_meanpool_ordered.csv", index_col=0)
db = db.rename(columns={"AUC": "DNABERT-2"})
db = (db.loc[category_4])["DNABERT-2"]

nt = pd.read_csv("ntv2_meanpool_ordered.csv", index_col=0)
nt = nt.rename(columns={"AUC": "NT-v2"})
nt = (nt.loc[category_4])["NT-v2"]

hyena = pd.read_csv("hyena_meanpool_ordered.csv", index_col=0)
hyena = hyena.rename(columns={"AUC": "HyenaDNA"})
hyena = (hyena.loc[category_4])["HyenaDNA"]

result = pd.concat([db, nt, hyena], axis=1)
result = result.round(3)
result.to_csv("./results_4/mean.csv")




# Multi-classification tasks, accuracy here used instead of AUC
# accuracy and summary token pooling
db = pd.read_csv("dnabert2_ordered.csv", index_col=0)
db = db.rename(columns={"Accuracy": "DNABERT-2"})
db = (db.loc[multi])["DNABERT-2"]

nt = pd.read_csv("ntv2_ordered.csv", index_col=0)
nt = nt.rename(columns={"Accuracy": "NT-v2"})
nt = (nt.loc[multi])["NT-v2"]

hyena = pd.read_csv("hyena_ordered.csv", index_col=0)
hyena = hyena.rename(columns={"Accuracy": "HyenaDNA"})
hyena = (hyena.loc[multi])["HyenaDNA"]

result = pd.concat([db, nt, hyena], axis=1)
result = result.round(3)
result.to_csv("./results_multi/summary_token.csv")

# accuracy and mean pooling
db = pd.read_csv("dnabert2_meanpool_ordered.csv", index_col=0)
db = db.rename(columns={"Accuracy": "DNABERT-2"})
db = (db.loc[multi])["DNABERT-2"]

nt = pd.read_csv("ntv2_meanpool_ordered.csv", index_col=0)
nt = nt.rename(columns={"Accuracy": "NT-v2"})
nt = (nt.loc[multi])["NT-v2"]

hyena = pd.read_csv("hyena_meanpool_ordered.csv", index_col=0)
hyena = hyena.rename(columns={"Accuracy": "HyenaDNA"})
hyena = (hyena.loc[multi])["HyenaDNA"]

result = pd.concat([db, nt, hyena], axis=1)
result = result.round(3)
result.to_csv("./results_multi/mean.csv")



# Mean Pooling vs Summary token
# dnabert2
mean = pd.read_csv("dnabert2_meanpool_ordered.csv", index_col=0)
mean = mean.rename(columns={"AUC": "Mean Pooling"})
mean = (mean.loc[order_list])["Mean Pooling"]

first = pd.read_csv("dnabert2_ordered.csv", index_col=0)
first = first.rename(columns={"AUC": "CLS Token"})
first = (first.loc[order_list])["CLS Token"]

result = pd.concat([mean, first], axis=1)
result = result.round(3)
result.to_csv("./results_pooling/dnabert2.csv")

# NT
mean = pd.read_csv("ntv2_meanpool_ordered.csv", index_col=0)
mean = mean.rename(columns={"AUC": "Mean Pooling"})
mean = (mean.loc[order_list])["Mean Pooling"]

first = pd.read_csv("ntv2_ordered.csv", index_col=0)
first = first.rename(columns={"AUC": "CLS Token"})
first = (first.loc[order_list])["CLS Token"]

result = pd.concat([mean, first], axis=1)
result = result.round(3)
result.to_csv("./results_pooling/ntv2.csv")

# Hyena
mean = pd.read_csv("hyena_meanpool_ordered.csv", index_col=0)
mean = mean.rename(columns={"AUC": "Mean Pooling"})
mean = (mean.loc[order_list])["Mean Pooling"]

first = pd.read_csv("hyena_ordered.csv", index_col=0)
first = first.rename(columns={"AUC": "EOS Token"})
first = (first.loc[order_list])["EOS Token"]

result = pd.concat([mean, first], axis=1)
result = result.round(3)
result.to_csv("./results_pooling/hyena.csv")

