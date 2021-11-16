

# table row=dimenstion coloumn=teams
from IPython.display import display
import math
import numpy as np
import pandas as pd

colors = [
    '#1f77b4',  # muted blue
    '#ff7f0e',  # safety orange
    '#2ca02c',  # cooked asparagus green
    '#d62728',  # brick red
    '#9467bd',  # muted purple
    '#8c564b',  # chestnut brown
    '#e377c2',  # raspberry yogurt pink
    '#7f7f7f',  # middle gray
    '#bcbd22',  # curry yellow-green
    '#17becf'   # blue-teal
]


def plot_spider(tbl):

    import plotly.graph_objects as go

    categories = tbl.index.values

    fig = go.Figure(layout=go.Layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    ))

    for i, c in enumerate(tbl.columns):
        fig.add_trace(go.Scatterpolar(
            r=tbl[c].values,
            theta=categories,
            fill='toself',
            name=c,
            opacity=0.6,
            marker=dict(color=colors[i])

        ))

    fig.update_layout(
        width=600, height=400,
        # template='ggplot2',
        template='gridon',
        polar=dict(
            # bgcolor='white',
            radialaxis=dict(
                visible=True,
                # color='black',
                # gridcolor='black',
                # angle=60,
                ticklen=2,
                tickangle=90,

                showtickprefix='first',
                range=[0, 1],
                tickvals=[0, .25, .5, .75, 1]
            ),
            gridshape='linear'
        ),
        showlegend=len(tbl.columns) > 1,

        font=dict(size=15, family="Arial Black")

    )

    fig.show()


def plot_multi_spider(dic_tbl):

    import math

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import pandas as pd

    cols = min(7, len(dic_tbl))

    rows = math.ceil(len(dic_tbl)/cols)

    fig = make_subplots(rows=rows, cols=cols,
                        subplot_titles=list(dic_tbl.keys()),
                        # font=10,
                        specs=[[{'type': 'polar'}]*cols]*rows,
                        figure=go.Figure(layout=go.Layout(
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)'
                        ))
                        )

    i = 0
    for claz in dic_tbl:
        tbl = pd.DataFrame(dic_tbl[claz])
        props = list(tbl.index)
        teams = list(tbl.columns)
        row = int(i/cols)+1
        col = i % cols+1
        for k, team in enumerate(teams):
            fig.add_trace(go.Scatterpolar(
                r=tbl[team].values,
                theta=props,
                fill='toself',
                name=team[0:15],
                opacity=0.6,
                legendgroup=team,
                showlegend=i == 0,
                # fillcolor='#551f77b4',#colors[k],
                marker=dict(color=colors[k])

            ), row=row, col=col)
            fig.update_polars(gridshape='linear',
                              # angularaxis_rotation=90,
                              row=row, col=col,
                              radialaxis=dict(

                                  visible=True,
                                  # color='black',
                                  # gridcolor='black',
                                  ticklen=2,
                                  tickangle=90,

                                  showtickprefix='first',
                                  range=[0, 1],
                                  tickvals=[0, .25, .5, .75, 1]
                              ))
        i += 1
    # fig.update_annotations(yshift=10)

    # fig.update_annotations(font_size=10)

    fig.update_layout(
        # width=600, height=400,
        template='gridon',
        polar=dict(
            # bgcolor='white',
            radialaxis=dict(
                visible=True,
                # color='black',
                # gridcolor='black',

                range=[0, 1],
            ),

            gridshape='linear'
        ),
        showlegend=len(tbl.columns) > 0,
        font=dict(size=15, family="Arial Black")

    )

    fig.show()


def display_dataset_info(gt):
    print('Dataset information==================')
    gt['duration'] = gt['offset']-gt['onset']
    total_dur = gt.iloc[-1]['offset']-gt.iloc[0]['onset']
    print(f"total duration of dataset: {total_dur:.0f} seconds")

    def percentile(n):
        def percentile_(x):
            return x.quantile(n)
        percentile_.__name__ = f'Q{n:.2f}'
        return percentile_

    def percentileSum(n):
        def percentile_(x):
            return x[x<=x.quantile(n)].sum()
        percentile_.__name__ = f'sum<Q{n:.2f}'
        return percentile_

    def dur_percent():
        def dur_percent_(x):
            return x.sum()/total_dur*100
        dur_percent_.__name__ = f'dur%'
        return dur_percent_

    infos = ['count', 'min', percentile(.1), percentile(.3), 'median', percentile(.7), percentile(.9),  'max', 'mean', 'std', percentileSum(.5), percentileSum(.7), percentileSum(.9), percentileSum(.95), 'sum', dur_percent()]
    res = gt.groupby(['event_label']).duration.agg(infos)
    res.loc['total'] = gt.duration.agg(infos)
    res2 = res.fillna(0).astype(np.int64)
    res2['dur%'] = res['dur%'].round(2)
    display(res2)

    # c_gt={c:groundtruth[groundtruth['event_label']==c] for c in allClass}
    # display(c_gt)
    # {'duration':pd.Timedelta(f"{groundtruth.iloc[-1]['offset']-groundtruth.iloc[0]['onset']:.0f}s"),
    #            'Number of concepts':len(groundtruth),
    # 'Concepts Info':'1'
    #           }
