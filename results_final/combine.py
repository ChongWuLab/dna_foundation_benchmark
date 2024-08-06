import pandas as pd
import os

def summarize_csv_files(directory_path, output_file):
    aggregated_df = pd.DataFrame()

    # Loop through all the files in the given directory
    for filename in sorted(os.listdir(directory_path)):
        # Construct the full file path
        file_path = os.path.join(directory_path, filename)

        df = pd.read_csv(file_path)
        # Pivot the DataFrame so that metrics become columns
        df_wide = df.set_index('Metric').T
        df_wide.reset_index(drop=True, inplace=True)

        # Add the file name (without extension) as a new column or index
        df_wide['Data'] = os.path.splitext(filename)[0]
        # Append to the aggregated DataFrame
        aggregated_df = pd.concat([aggregated_df, df_wide], ignore_index=True)


    # Reorder the DataFrame to have 'Data' as the first column
    cols = ['Data'] + [col for col in aggregated_df.columns if col != 'Data']
    aggregated_df = aggregated_df[cols]
    aggregated_df.set_index('Data', inplace=True)

    aggregated_df.to_csv(output_file)


summarize_csv_files('./dnabert2', 'final_dnabert2.csv')
summarize_csv_files('./dnabert2_meanpool', 'final_dnabert2_meanpool.csv')
summarize_csv_files('./ntv2', 'final_ntv2.csv')
summarize_csv_files('./ntv2_meanpool', 'final_ntv2_meanpool.csv')
summarize_csv_files('./hyena_meanpool', 'final_hyena_meanpool.csv')
summarize_csv_files('./hyena', 'final_hyena.csv')