import marimo

__generated_with = "0.10.6"
app = marimo.App(width="medium", auto_download=["ipynb", "html"])


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import pandas as pd
    import plotly.graph_objects as go
    import numpy
    import plotly.express as px
    from sklearn.ensemble import RandomForestRegressor
    from plotly.subplots import make_subplots
    import statsmodels.api as sm
    from sklearn.preprocessing import OneHotEncoder
    return (
        OneHotEncoder,
        RandomForestRegressor,
        go,
        make_subplots,
        mo,
        numpy,
        pd,
        pl,
        px,
        sm,
    )


@app.cell
def _(mo):
    mo.md(r"""# What are the factors affecting rent prices for HDB flats in Singapore?""")
    return


@app.cell
def _(pl):
    df = pl.read_csv("RentingOutofFlats2024CSV.csv", infer_schema_length=10000)
    return (df,)


@app.cell
def _(df):
    df.describe()
    return


@app.cell
def _(mo):
    mo.md(r"""no Null imputation needed, column names are consistent""")
    return


@app.cell
def _(df):
    df.is_duplicated().any()
    return


@app.cell
def _(df):
    df["town"].unique().to_list()
    return


@app.cell
def _(df):
    df["flat_type"].unique().to_list()
    return


@app.cell
def _(mo):
    mo.md(r"""probably consistent values in the features. checking from the charts below also paints a similar stories""")
    return


@app.cell
def _(df, pl):
    df2 = df.with_columns(
        pl.col("rent_approval_date").str.to_datetime("%Y-%m")
    ).unique()  # removes duplicates
    return (df2,)


@app.cell
def _(mo):
    mo.md(r"""# box plots""")
    return


@app.cell(hide_code=True)
def _(df2, px):
    px.box(
        df2,
        x="town",
        y="monthly_rent",
    )
    return


@app.cell
def _(mo):
    mo.md(r"""relatively similar distribution between many of the towns. hard to see really patterns with this chart""")
    return


@app.cell(hide_code=True)
def _(df2, px):
    px.box(
        df2,
        x="flat_type",
        y="monthly_rent",
        category_orders={
            "flat_type": [
                "1-ROOM",
                "2-ROOM",
                "3-ROOM",
                "4-ROOM",
                "5-ROOM",
                "EXECUTIVE",
            ]
        },
    )
    return


@app.cell
def _(mo):
    mo.md("""2 room and 3 rooms flats has similar distribution. 4,5 rooms and executive have similar distribution. median increases by the number of rooms.""")
    return


@app.cell
def _():
    # _df = df2.group_by("town").median().sort("monthly_rent")
    # _df2 = (
    #     df2.group_by("town").len().sort("len").rename({"len": "number_of_units"})
    # )

    # town_sorted2 = _df["town"]
    # px.bar(
    #     _df,
    #     "town",
    #     "monthly_rent",
    # )
    return


@app.cell
def _(mo):
    mo.md(r"""# bar and line charts""")
    return


@app.cell(hide_code=True)
def _(df2, go, make_subplots):
    _df2 = (
        df2.group_by("town")
        .len()
        .rename({"len": "number_of_units"})
        .sort("number_of_units")
        .join(
            df2.group_by("town").median().select("town", "monthly_rent"), on="town"
        )
    ).sort("monthly_rent")
    _df2 = _df2


    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            x=_df2["town"],
            y=_df2["monthly_rent"],
            name="monthly_rent",
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=_df2["town"],
            y=_df2["number_of_units"],
            name="number_of_units",
            mode="lines",
            line=dict(
                color="RGBA(255, 165, 0, 1)",
            ),
        ),
        secondary_y=True,
    )
    fig
    return (fig,)


@app.cell(disabled=True, hide_code=True)
def _(df2, px, town_sorted2):
    _df = df2.group_by("town").len().sort("len").rename({"len": "number_of_units"})
    px.bar(
        _df,
        "town",
        "number_of_units",
        category_orders={"town": town_sorted2},
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        town seems to have effect on median monthly_rent prices. Central,bukit timah,bukit merah,bishan,queens, are the most expensives. There is probably alot of demand for these towns.

        here are not much correlations betweeen number_of_units to monthly_rent across town.
        """
    )
    return


@app.cell
def _():
    # _df = df2.group_by("flat_type").median().sort("monthly_rent")
    # px.bar(
    #     _df,
    #     "flat_type",
    #     "monthly_rent",
    # )
    return


@app.cell
def _(mo):
    mo.md(r""" """)
    return


@app.cell(hide_code=True)
def _(df2, go, make_subplots):
    _df2 = (
        df2.group_by("flat_type")
        .len()
        .rename({"len": "number_of_units"})
        .sort("number_of_units")
        .join(
            df2.group_by("flat_type").median().select("flat_type", "monthly_rent"),
            on="flat_type",
        )
    ).sort("flat_type")
    # print(_df2)


    _fig = make_subplots(specs=[[{"secondary_y": True}]])
    _fig.add_trace(
        go.Bar(
            x=_df2["flat_type"],
            y=_df2["monthly_rent"],
            name="monthly_rent",
        ),
        secondary_y=False,
    )

    _fig.add_trace(
        go.Scatter(
            x=_df2["flat_type"],
            y=_df2["number_of_units"],
            name="number_of_units",
            mode="lines",
        ),
        secondary_y=True,
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
        flat types seems to have effect on median rent prices. the more rooms, the more expensive

        There are some non linear correlations betweeen number_of_units to monthly_rent across flat_types.


        street_name and block are not needed as town feature would b enough information about it's correlations to monthly_rent
        """
    )
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _():
    # px.scatter(
    #     df2.group_by("flat_type", "town").median(),
    #     "town",
    #     "flat_type",
    #     size="monthly_rent",
    #     color="monthly_rent",
    #     category_orders={
    #         "flat_type": [
    #             "1-ROOM",
    #             "2-ROOM",
    #             "3-ROOM",
    #             "4-ROOM",
    #             "5-ROOM",
    #             "EXECUTIVE",
    #         ],
    #         "town": town_sorted,
    #     },
    #     size_max=20,
    # ).update_layout(height=600, width=1000)
    return


@app.cell
def _(mo):
    mo.md(r"""<!-- across flat_typeflat_type and towntown monthly_rentmonthly_rent does increases for the more premium flat-typeflat-type and in certain townstowns. -->""")
    return


@app.cell(hide_code=True)
def _():
    # px.scatter(
    #     df2.group_by("flat_type", "town").len(),
    #     "town",
    #     "flat_type",
    #     size="len",
    #     category_orders={
    #         "flat_type": [
    #             "1-ROOM",
    #             "2-ROOM",
    #             "3-ROOM",
    #             "4-ROOM",
    #             "5-ROOM",
    #             "EXECUTIVE",
    #         ],
    #         "town": town_sorted,
    #     },
    #     size_max=20,
    # ).update_layout(height=600, width=1000)
    return


@app.cell
def _(mo):
    mo.md(r"""<!-- comparing the two bubble plots aboves that distinguishes in terms of flat_typeflat_type and towntown seems to indicate that number_of_unitsnumber_of_units available may have some correlations to monthly_rentmonthly_rent but there are exceptions such as jurong eastjurong east and sengkangsengkang. -->""")
    return


@app.cell
def _(df2, px):
    px.scatter(
        df2.sample(df2.shape[0] / 3),
        x="rent_approval_date",
        y="monthly_rent",
        trendline="lowess",
    )
    return


@app.cell
def _(mo):
    mo.md("""rent approval rates has a slight positive correlation to monthly rent. especially after july 2022""")
    return


@app.cell
def _(mo):
    mo.md(r"""# Hypothesis testing""")
    return


@app.cell
def _(OneHotEncoder, df2, pl):
    one_hot_encoded_x = (
        OneHotEncoder(sparse_output=False)
        .set_output(transform="polars")
        .fit_transform(
            df2.select(pl.col(pl.String()))
            .select(pl.exclude("street_name", "block"))
            .to_pandas()
        )
    )
    return (one_hot_encoded_x,)


@app.cell
def _(df2, one_hot_encoded_x, pl):
    df3 = df2.select(pl.exclude(pl.String())).hstack(one_hot_encoded_x)
    return (df3,)


@app.cell
def _(df3, pl, sm):
    x = (
        df3.select(pl.exclude("monthly_rent"))
        .with_columns(pl.col("rent_approval_date").cast(pl.Int64))
        # .with_columns(
        #     rent_approval_date=pl.col("rent_approval_date")
        #     / pl.col(
        #         "rent_approval_date"
        #     ).median()  # trying to reduce these numbers into something smaller
        # )
        .to_pandas()
    )
    y = df3.select("monthly_rent").to_numpy()
    # print(_x)

    _x = sm.add_constant(x)
    results = sm.OLS(y, _x).fit()

    results.summary()
    return results, x, y


@app.cell
def _(mo):
    mo.md(r"""all features are statistically significant (alpha =0.05)""")
    return


@app.cell
def _(RandomForestRegressor, x, y):
    rfr = RandomForestRegressor(n_jobs=20)
    rfr.fit(x, y.ravel())
    return (rfr,)


@app.cell
def _(pl, rfr, x):
    pl.DataFrame(
        [x.columns.tolist(), rfr.feature_importances_.tolist()],
        schema=["feature", "importance"],
    ).sort("importance", descending=True).head(10)
    return


@app.cell
def _(mo):
    mo.md(r"""rent_approval_date has the highest importance. flat type 2,3,4 are in the top ten. places with higher rent are in bukit merah, queenstown and central. overall expected correlated features are here in the importances hierachy.""")
    return


@app.cell
def _(df3, pl, px):
    _d = df3.corr().with_columns(pl.all().round(2))
    px.imshow(_d, text_auto=True).update_yaxes(
        tickmode="array",
        ticktext=df3.columns,
        tickvals=[x for x in range(df3.shape[1])],
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        monthly_rent is highly corelated to rent_approval_date not sure what to do with this information.


        # overall flat_type, rend_approval_dates affects monthly_rent the most, it is back by their statistic significances and feature importances.
        """
    )
    return


if __name__ == "__main__":
    app.run()
