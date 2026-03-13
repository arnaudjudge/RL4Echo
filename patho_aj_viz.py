import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np

def wrap_label(text, width=12):
    """
    Insert <br> after approximately `width` characters,
    breaking only at spaces, preserving full words.
    """
    import textwrap
    if not isinstance(text, str):
        return text
    return "<br>".join(textwrap.wrap(text, width=width, break_long_words=False))


if __name__ == "__main__":
    df = pd.read_csv('patho_AJ_abbr.csv')
    df.loc[df['Proposed category'] == 'Shape anomaly – other', 'Proposed category'] = 'Shape anomaly'
    df.loc[df['Name'] == 'Tricuspid Valvular Regurgitation ', 'Name'] = 'Tricuspid Regurgitation'
    df.loc[df['Name'] == 'Mitral Valvular Regurgitation ', 'Name'] = 'Mitral Regurgitation'
    df.loc[df['Proposed category'] == 'Conduction problem', 'Proposed category'] = 'Conduction'
    df["nb_all_rescaled"] = np.sqrt(df["nb_all"])
    df_log = df.copy()
    df_log["nb_all_log"] = np.log1p(df["nb_all"])
    alpha = 0.2
    df["nb_all_soft"] = (df["nb_all"] ** alpha)
    df["nb_test_soft"] = (df["nb_test"] ** alpha)


    df2 = df.copy()
    df2["Name_abbr_wrapped"] = df2["Name_abbr"].apply(lambda s: wrap_label(s, width=12))
    df2["Chamber / structure wrapped"] = df2["Chamber / structure"].apply(lambda s: wrap_label(s, width=8))

    # counts = df['Proposed category'].value_counts().plot.pie(
    #     autopct='%1.1f%%', startangle=90,
    #     wedgeprops={'edgecolor': 'black'}, legend=False)
    #
    # # Optional: Apply Seaborn style for a more visually appealing plot
    # sns.set_style("whitegrid")  # Or "darkgrid", "white", "dark", "ticks"
    #
    # plt.title('Distribution of Categories')
    # plt.ylabel('')  # Hide the default 'Value' label on the y-axis
    # plt.show()



    fig = px.sunburst(
        df2,
        path=["Proposed category", "Chamber / structure wrapped", "Name_abbr_wrapped"],
        values="nb_all_soft",
        color="Proposed category",
        color_discrete_sequence=[
            "#f8d377",  # "#FED0AF",
            "#feac72",  # "#fbe6b1",
            "#a0cfa7", #"#c9e4cd",
            "#c6a9b0",  # "#deced2",
            "#82c4ed", #"#b8def5",
        ],
        title="Sunburst with forced thicker outer ring",
    )

    # fig = px.sunburst(
    #     df,
    #     path=["Proposed category", "Chamber / structure", "Name_abbr"],
    #     values="nb_test_soft",  # or "nb_test" or None if you just want counts
    #     color="Proposed category",
    #     title="Sunburst of Chambers, Categories, and Names",
    # )

    fig.update_layout(margin=dict(t=50, l=10, r=10, b=10))
    fig.show()