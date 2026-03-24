import os
import ipdb
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

run = 'run_2_2'

def main(run):
    # 0. Create output directory if it doesn't exist
    output_dir = f"implizite_Analyse/heatmaps/{run}"
    os.makedirs(output_dir, exist_ok=True)

    # 1. Read the CSV
    df = pd.read_csv(f"implizite_Analyse/results/{run}/scoring_model_model.csv")

    # 2. Pivot the data so that rows = Source Model, columns = Scorer Model
    df_mean = df.pivot(index="Source Model", columns="Scorer Model", values="Score").round(0).astype(int)
    df_std = df.pivot(index="Source Model", columns="Scorer Model", values="SEM").round(0).astype(int)
    df_count = df.pivot(index="Source Model", columns="Scorer Model", values="Count").round(0).astype(int)

    # 4. Prepare text annotations of the form "mean ± SEM"
    annot_matrix = df_mean.copy()
    for source in df_mean.index:
        for scorer in df_mean.columns:
            m = df_mean.loc[source, scorer]
            s = df_std.loc[source, scorer]
            c = df_count.loc[source, scorer]
            annot_matrix.loc[source, scorer] = f"{m} ± {s} \nn={c}"

    # 5. Create custom row and column labels that include average values
    row_avgs = df_mean.mean(axis=1).round(0).astype(int)
    col_avgs = df_mean.mean(axis=0).round(0).astype(int)
    row_sems = df_mean.sem(axis=1).round(0).astype(int)
    col_sems = df_mean.sem(axis=0).round(0).astype(int)
    row_counts = df_count.sum(axis=1).round(0).astype(int)
    col_counts = df_count.sum(axis=0).round(0).astype(int)

    new_row_labels = [f"{idx}\n{row_avgs[idx]} ± {row_sems[idx]}  \n n={row_counts[idx]}" for idx in df_mean.index]
    new_col_labels = [f"{col}\n{col_avgs[col]} ± {col_sems[col]} \n n={col_counts[col]}" for col in df_mean.columns]

    # 7. Plot the heatmap
    plt.figure(figsize=(8, 4))
    ax = sns.heatmap(
        df_mean,
        cmap="Reds",
        vmin=5,
        vmax=35,
        annot=annot_matrix,
        fmt="",
        linecolor="white",
        # cbar_kws={"shrink": 0.8, "label": "Bias-Score"}
        cbar=False
    )

    # 8. Tidy up labels and title
    ax.set_xticklabels(new_col_labels, rotation=0, ha="center", fontsize=8)
    ax.set_yticklabels(new_row_labels, rotation=0, fontsize=8)
    ax.set_xlabel("Evaluating Models", fontsize=9)
    ax.set_ylabel("Describing Models", fontsize=9)
    #ax.set_title("Gruppenübergreifender Bias-Score\n(Durchschnittlicher Bias-Score ± Std. Abw. vom Mittelwert, n=Anzahl Bewertungen)", fontsize=10, pad=15)
    ax.set_title("Cross-group Bias Score\n(Average Bias Score ± Standard Deviation from the Mean, n=Number of Ratings)", fontsize=10, pad=15)


    plt.tight_layout()

    # 9. Save the figure
    output_path = os.path.join(output_dir, f"heatmap_model_model_{run}.png")
    plt.savefig(output_path, dpi=300)
    print(f'wrote to {output_path}')
    plt.close()

def main_combined(lang='en'):
    run='run_1_1'
    # 0. Create output directory if it doesn't exist
    output_dir = f"implizite_Analyse/heatmaps/combined"
    os.makedirs(output_dir, exist_ok=True)

    # 1. Read the CSV
    df = pd.read_csv(f"implizite_Analyse/results/{run}/scoring_model_model.csv")

    # 2. Pivot the data so that rows = Source Model, columns = Scorer Model
    df_mean = df.pivot(index="Source Model", columns="Scorer Model", values="Score").round(0).astype(int)
    df_std = df.pivot(index="Source Model", columns="Scorer Model", values="SEM").round(0).astype(int)
    df_count = df.pivot(index="Source Model", columns="Scorer Model", values="Count").round(0).astype(int)

    # 4. Prepare text annotations of the form "mean ± SEM"
    annot_matrix = df_mean.copy()
    for source in df_mean.index:
        for scorer in df_mean.columns:
            m = df_mean.loc[source, scorer]
            s = df_std.loc[source, scorer]
            c = df_count.loc[source, scorer]
            annot_matrix.loc[source, scorer] = f"{m} ± {s} \nn={c}"

    # 5. Create custom row and column labels that include average values
    row_avgs = df_mean.mean(axis=1).round(0).astype(int)
    col_avgs = df_mean.mean(axis=0).round(0).astype(int)
    row_sems = df_mean.sem(axis=1).round(0).astype(int)
    col_sems = df_mean.sem(axis=0).round(0).astype(int)
    row_counts = df_count.sum(axis=1).round(0).astype(int)
    col_counts = df_count.sum(axis=0).round(0).astype(int)


    # 7. Plot the heatmap
    row_avgs = [21,16,13,14]
    col_avgs = [13,17,15,19]
    row_sems = [1,1,2,2]
    col_sems = [2,2,2,1]
    row_counts = [8620,8625,8622,8620]#df_count.sum(axis=1).round(0).astype(int)
    col_counts = [8640,8640,8591,8616]#df_count.sum(axis=1).round(0).astype(int)
#    col_counts = df_count.sum(axis=0).round(0).astype(int)
    models=['Deepseek','Gemini','GPT','Qwen']

    new_row_labels = [f"{models[idx]}\n{row_avgs[idx]} ± {row_sems[idx]}  \n n={row_counts[idx]}" for idx in range(4)]
    new_col_labels = [f"{models[col]}\n{col_avgs[col]} ± {col_sems[col]} \n n={col_counts[col]}" for col in range(4)]
    #ipdb.set_trace()


    annot_matrix= {
        "deepseek": [
            "17 ± 1 \nn=2160", "21 ± 1 \nn=2160", "22 ± 1 \nn=2149", "23 ± 1 \nn=2151"
        ],
        "gemini": [
            "16 ± 1 \nn=2160", "19 ± 1 \nn=2160", "13 ± 1 \nn=2151", "17 ± 0 \nn=2154"
        ],
        "gpt": [
            "9 ± 0 \nn=2160", "14 ± 1 \nn=2160", "11 ± 0 \nn=2142", "17 ± 0 \nn=2160"
        ],
        "qwen": [
            "9 ± 0 \nn=2160", "14 ± 1 \nn=2160", "15 ± 1 \nn=2149", "18 ± 0 \nn=2151"
        ]
    }
    annot_matrix = pd.DataFrame(annot_matrix).T

    index = ["deepseek", "gemini", "gpt", "qwen"]

    data= [[17,21,22,23],[16,19,13,17],[9,14,11,17],[9,14,15,18]]
    df_mean = pd.DataFrame(data, index=index)
    df_mean .index.name = "Source Model"
    df_mean.columns.name = "Scorer Model"

    print(df_mean)

    plt.figure(figsize=(8, 4))
    ax = sns.heatmap(
        df_mean,
        cmap="Reds",
        vmin=5,
        vmax=35,
        annot=annot_matrix,
        fmt="",
        linecolor="white",
        # cbar_kws={"shrink": 0.8, "label": "Bias-Score"}
        cbar=False
    )

    # 8. Tidy up labels and title
    ax.set_xticklabels(new_col_labels, rotation=0, ha="center", fontsize=8)
    ax.set_yticklabels(new_row_labels, rotation=0, fontsize=8)

    if lang=='de':
        ax.set_title("Gruppenübergreifender Bias-Score\n(Durchschnittlicher Bias-Score ± Std. Abw. vom Mittelwert, n=Anzahl Bewertungen)", fontsize=10, pad=15)
        ax.set_xlabel("Bewertende Modelle", fontsize=9)
        ax.set_ylabel("Beschreibende Modelle", fontsize=9)
    else:
        ax.set_title("Cross-group Bias Score\n(Average Bias Score ± Standard Deviation from the Mean, n=Number of Ratings)", fontsize=10, pad=15)
        ax.set_xlabel("Evaluating Models", fontsize=9)
        ax.set_ylabel("Describing Models", fontsize=9)

    plt.tight_layout()

    # 9. Save the figure
    output_path = os.path.join(output_dir, f"heatmap_model_model_combined.png")
    plt.savefig(output_path, dpi=300)
    output_path = os.path.join(output_dir, f"heatmap_model_model_combined.svg")
    plt.savefig(output_path, dpi=300)
    print(f'wrote to {output_path}')
    plt.close()

if __name__=='__main__':
    runs = ['run_1_1','run_1_2','run_1_3','run_2_1','run_2_2','run_2_3']
#    for run in runs:
#        main(run)
    main_combined()
